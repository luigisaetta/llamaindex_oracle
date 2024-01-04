#
# Translator from English to Italian
#
from oci.ai_language import AIServiceLanguageClient
from oci.ai_language.models import TextDocument
from oci.ai_language.models import BatchLanguageTranslationDetails

from typing import List

from config_private import COMPARTMENT_OCID


class OCITranslator:
    def __init__(self, oci_config, source_language="en", target_language="it"):
        # define the client
        self.ai_client = AIServiceLanguageClient(oci_config)
        self.source_language = source_language
        self.target_language = target_language

    def translate(self, texts: List[str]):
        # key is fake
        txt_docs = [
            TextDocument(text=text, key=f"key{i}", language_code=self.source_language)
            for i, text in enumerate(texts)
        ]

        batch_lan_transl_details = BatchLanguageTranslationDetails(
            documents=txt_docs,
            compartment_id=COMPARTMENT_OCID,
            target_language_code=self.target_language,
        )

        # here we call to translate the batch
        txt_docs_transl = self.ai_client.batch_language_translation(
            batch_lan_transl_details
        ).data

        return txt_docs_transl
