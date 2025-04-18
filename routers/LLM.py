from azure.identity import DefaultAzureCredential
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from dotenv import load_dotenv
import os


def get_llm():
    """ Function to initialize the LLM so that it can be used in the app """
    # Set the useLocalLLM and useAzureOpenAI variables based on environment variables
    useLocalLLM = os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"
    useAzureOpenAI = os.environ.get("USE_AZURE_OPENAI", "false").lower() == "true"

    # if both are True, raise error
    if useLocalLLM and useAzureOpenAI:
        raise Exception("USE_LOCAL_LLM and USE_AZURE_OPENAI cannot both be true")

    kernel = None
    endpoint = os.environ.get("AI_ENDPOINT") or os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    useAzureAD = os.environ.get("USE_AZURE_AD", "false").lower() == "true"

    if not (useLocalLLM or useAzureOpenAI):
        if not api_key:
            raise Exception("OPENAI_API_KEY is required when not using Azure OpenAI")
        org_id = os.environ.get("OPENAI_ORG_ID")
        if not org_id:
            raise Exception("OPENAI_ORG_ID is required when not using Azure OpenAI")

    kernel = sk.Kernel()

    if useLocalLLM:
        print("Using Local LLM (not configured in this sample)")
    elif useAzureOpenAI:
        print("Using Azure OpenAI")
        if not endpoint or not deployment:
            raise Exception("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT_NAME are required")
        if useAzureAD:
            print("Authenticating with Azure AD")
            credential = DefaultAzureCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/.default").token
            kernel.add_chat_service(
                "dv",
                AzureChatCompletion(deployment_name=deployment, endpoint=endpoint, ad_token=token)
            )
        else:
            print("Authenticating with OpenAI API key (Azure OpenAI)")
            if not api_key:
                raise Exception("OPENAI_API_KEY is required")
            kernel.add_chat_service(
                "dv",
                AzureChatCompletion(deployment_name=deployment, endpoint=endpoint, api_key=api_key)
            )
    else:
        print("Using OpenAI (non-Azure)")
        kernel.add_chat_service(
            "dv",
            OpenAIChatCompletion(model_id="gpt-3.5-turbo", api_key=api_key, org_id=org_id)
        )

    return kernel, useLocalLLM, endpoint
