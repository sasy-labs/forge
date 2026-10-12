"""
Observability and RM setup

Authentication is configured via environment variables:

    AUTH_METHOD: Authentication method to use. Options:
        - "none": No authentication (default)
        - "api_key": API key authentication
        - "client_credentials": OAuth2 client credentials grant
        - "password": OAuth2 resource owner password grant
        - "browser": OAuth2 authorization code with PKCE (opens browser)
        - "device": OAuth2 device authorization grant
        - "mtls": Mutual TLS (client certificates)

    For API key (AUTH_METHOD=api_key):
        SASY_API_KEY: The API key to use

    For OAuth2 methods (client_credentials, password, browser, device):
        OIDC_ISSUER_URL: OIDC issuer URL
            Default: https://localhost:8443/realms/malade
        OIDC_CLIENT_ID: OAuth2 client ID
            Default: malade-agents (confidential) or malade-cli (public)
        OIDC_CLIENT_SECRET: OAuth2 client secret
            Default: malade-agents-secret (for confidential clients)
        OIDC_CA_CERT: Path to CA certificate for TLS verification
            Default: ../../../../certs/ca.crt (if exists)

    For password grant (AUTH_METHOD=password):
        OIDC_USERNAME: Username
        OIDC_PASSWORD: Password

Local development: Just set AUTH_METHOD=client_credentials - defaults work with local Keycloak.
"""

import logging
import langroid as lr
import sasy.instrumentation as instrumentation
import os
from sasy.auth.hooks import (
    NoAuthHook,
    APIKeyAuthHook,
    OIDCAuthHook,
    BrowserAuthHook,
    DeviceAuthHook,
)
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

# Load environment variables before any configuration
load_dotenv()
load_dotenv(".env-credentials", override=True)


def get_auth_hook():
    """Create auth hook based on AUTH_METHOD environment variable.

    Defaults to api_key auth when SASY_API_KEY is set.
    """
    api_key = os.environ.get("SASY_API_KEY")
    default_method = "api_key" if api_key else "none"
    method = os.environ.get("AUTH_METHOD", default_method).lower()

    if method == "none":
        logger.info("Auth method: none")
        return NoAuthHook()

    if method == "api_key":
        if not api_key:
            raise ValueError("SASY_API_KEY required for api_key auth")
        logger.info("Auth method: api_key")
        return APIKeyAuthHook(api_key=api_key)

    if method == "mtls":
        # mTLS is handled by TLS config, no auth hook needed
        logger.info("Auth method: mtls (handled by TLS config)")
        return NoAuthHook()

    # OAuth2 methods - defaults for local Keycloak development
    issuer_url = os.environ.get(
        "OIDC_ISSUER_URL", "https://localhost:8443/realms/malade"
    )

    # Default client IDs based on auth method
    if method in ("client_credentials", "password"):
        default_client_id = "malade-service"
        default_client_secret = "malade-service-secret"
    else:
        default_client_id = "malade"
        default_client_secret = None

    client_id = os.environ.get("OIDC_CLIENT_ID", default_client_id)
    client_secret = os.environ.get("OIDC_CLIENT_SECRET", default_client_secret)

    # CA cert for trusting Keycloak's TLS certificate
    # Default to local development CA cert location
    default_ca_cert = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "certs", "ca.crt"
    )
    if os.path.exists(default_ca_cert):
        default_ca_cert = os.path.abspath(default_ca_cert)
    else:
        default_ca_cert = None
    ca_cert = os.environ.get("OIDC_CA_CERT", default_ca_cert)
    scope = os.environ.get("OIDC_SCOPE", "openid profile fda-access openai-access")

    issuer_url = issuer_url.rstrip("/")
    token_url = f"{issuer_url}/protocol/openid-connect/token"
    auth_url = f"{issuer_url}/protocol/openid-connect/auth"
    device_auth_url = f"{issuer_url}/protocol/openid-connect/auth/device"

    if method == "client_credentials":
        logger.info(f"Auth method: client_credentials (issuer: {issuer_url})")
        return OIDCAuthHook(
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret,
            scope=scope,
            ca_cert=ca_cert,
        )

    if method == "password":
        username = os.environ.get("OIDC_USERNAME")
        password = os.environ.get("OIDC_PASSWORD")
        if not username or not password:
            raise ValueError("OIDC_USERNAME and OIDC_PASSWORD required for password auth")
        logger.info(f"Auth method: password (issuer: {issuer_url}, user: {username})")
        return OIDCAuthHook(
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret,
            username=username,
            password=password,
            scope=scope,
            ca_cert=ca_cert,
        )

    if method == "browser":
        redirect_port = int(os.environ.get("OIDC_REDIRECT_PORT", "8400"))
        logger.info(f"Auth method: browser/PKCE (issuer: {issuer_url})")
        return BrowserAuthHook(
            authorization_url=auth_url,
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret,
            scope=scope,
            redirect_port=redirect_port,
            ca_cert=ca_cert,
        )

    if method == "device":
        logger.info(f"Auth method: device (issuer: {issuer_url})")
        return DeviceAuthHook(
            device_authorization_url=device_auth_url,
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret,
            scope=scope,
            ca_cert=ca_cert,
        )

    raise ValueError(
        f"Unknown AUTH_METHOD: {method}. "
        "Valid options: none, api_key, client_credentials, password, browser, device, mtls"
    )


# Feedback callback to append authorization feedback to responses
def append_feedback_to_response(accumulator, output, agent):
    """Append authorization feedback to the agent's response."""
    if accumulator.has_denials():
        for d in accumulator.get_denials():
            # d.message is "HTTP request to <url>: <deny_message>"; extract just the reason
            reason = d.message.rsplit(": ", 1)[-1] if ": " in d.message else d.message
            print(f"[POLICY] DENIED — {reason}")
        feedback_msg = accumulator.format_for_llm()
        if output is not None:
            original = output.content or ""
            output.content = original + f"\n\n{feedback_msg}"


# Configure auth hook and instrumentation settings
_auth_hook = get_auth_hook()
_log_policy = int(os.environ.get("LOG_POLICY_DECISIONS", "0")) > 0
_log_transforms = int(os.environ.get("LOG_TRANSFORMS", "0")) > 0
instrumentation.configure(
    auth_hook=_auth_hook,
    log_denials=True,
    log_policy_decisions=_log_policy,
    log_policy_decisions_transforms_only=_log_policy,
    log_transforms=_log_transforms,
    feedback_callback=append_feedback_to_response,
)

# Instrument Langroid agents and HTTP libraries (HTTP routing, which checks
# requests and injects credentials, is off unless asked for)
instrumentation.instrument(http=True)

# Fix requirement for OpenAI API key in embedding models
def allow_missing_key_wrapper(wrapped, instance, args, kwargs):
    """Ensure that we can use embeddings without API keys."""
    self = instance
    super(lr.embedding_models.OpenAIEmbeddings, self).__init__()
    config = args[0] if args else kwargs.get(
        "config", lr.embedding_models.OpenAIEmbeddingsConfig()
    )
    self.config = config
    load_dotenv()

    # Check if using LangDB
    self.is_langdb = self.config.model_name.startswith("langdb/")

    if self.is_langdb:
        self.config.model_name = self.config.model_name.replace("langdb/", "")
        self.config.api_base = self.config.langdb_params.base_url
        project_id = self.config.langdb_params.project_id
        if project_id:
            self.config.api_base += "/" + project_id + "/v1"
        self.config.api_key = self.config.langdb_params.api_key

    if not self.config.api_key:
        self.config.api_key = os.getenv("OPENAI_API_KEY", "")

    self.config.organization = os.getenv("OPENAI_ORGANIZATION", "")

    # Drop key check

    self.client = OpenAI(
        base_url=self.config.api_base,
        api_key=self.config.api_key,
        organization=self.config.organization,
    )
    model_for_tokenizer = self.config.model_name
    if model_for_tokenizer.startswith("openai/"):
        self.config.model_name = model_for_tokenizer.replace("openai/", "")
    self.tokenizer = tiktoken.encoding_for_model(self.config.model_name)


wrap_function_wrapper(
    "langroid.embedding_models.models",
    "OpenAIEmbeddings.__init__",
    allow_missing_key_wrapper,
)
