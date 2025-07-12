from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator
from datasets import load_dataset
from haystack.dataclasses import ChatMessage
from typing import Optional, List, Union, Dict
from .config import DatasetConfig, DATASET_CONFIGS, MODEL_CONFIG

class RAGPipeline:
    def __init__(
        self,
        google_api_key: str,
        dataset_config: Union[str, DatasetConfig],
        documents: Optional[List[Union[str, Document]]] = None,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None
    ):
        """
        Initialize the RAG Pipeline.
        
        Args:
            google_api_key: API key for Google AI services
            dataset_config: Either a string key from DATASET_CONFIGS or a DatasetConfig object
            documents: Optional list of documents to use instead of loading from a dataset
            embedding_model: Optional override for embedding model
            llm_model: Optional override for LLM model
        """
        # Load configuration
        if isinstance(dataset_config, str):
            if dataset_config not in DATASET_CONFIGS:
                raise ValueError(f"Dataset config '{dataset_config}' not found. Available configs: {list(DATASET_CONFIGS.keys())}")
            self.config = DATASET_CONFIGS[dataset_config]
        else:
            self.config = dataset_config

        # Load documents either from provided list or dataset
        if documents is not None:
            self.documents = documents
        else:
            dataset = load_dataset(self.config.name, split=self.config.split)
            # Create documents with metadata based on configuration
            self.documents = []
            for doc in dataset:
                # Create metadata dictionary from configured fields
                meta = {}
                if self.config.fields:
                    for meta_key, dataset_field in self.config.fields.items():
                        if dataset_field in doc:
                            meta[meta_key] = doc[dataset_field]
                
                # Create document with content and metadata
                document = Document(
                    content=doc[self.config.content_field],
                    meta=meta
                )
                self.documents.append(document)

        # print 10 documents
        for doc in self.documents[:10]:
            print(f"Content: {doc.content}")
            print(f"Metadata: {doc.meta}")
            print("-"*100)
        
        # Initialize components
        self.document_store = InMemoryDocumentStore()
        self.doc_embedder = SentenceTransformersDocumentEmbedder(
            model=embedding_model or MODEL_CONFIG["embedding_model"]
        )
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=embedding_model or MODEL_CONFIG["embedding_model"]
        )
        self.retriever = InMemoryEmbeddingRetriever(self.document_store)
        
        # Warm up the document embedder
        self.doc_embedder.warm_up()
        
        # Initialize prompt template
        template = [
            ChatMessage.from_user(self.config.prompt_template)
        ]
        self.prompt_builder = ChatPromptBuilder(template=template)

        # Initialize the generator
        self.generator = GoogleAIGeminiChatGenerator(
            model=llm_model or MODEL_CONFIG["llm_model"]
        )
        
        # Index documents
        self._index_documents(self.documents)
        
        # Build pipeline
        self.pipeline = self._build_pipeline()

    @classmethod
    def from_preset(cls, google_api_key: str, preset_name: str):
        """
        Create a pipeline from a preset configuration.
        
        Args:
            google_api_key: API key for Google AI services
            preset_name: Name of the preset configuration to use
        """
        return cls(google_api_key=google_api_key, dataset_config=preset_name)

    def _index_documents(self, documents):
        # Embed and index documents
        docs_with_embeddings = self.doc_embedder.run(documents)
        self.document_store.write_documents(docs_with_embeddings["documents"])
    
    def _build_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("text_embedder", self.text_embedder)
        pipeline.add_component("retriever", self.retriever)
        pipeline.add_component("prompt_builder", self.prompt_builder)
        pipeline.add_component("llm", self.generator)
        
        # Connect components
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever", "prompt_builder")
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        
        return pipeline
    
    def answer_question(self, question: str) -> str:
        """Run the RAG pipeline to answer a question"""
        result = self.pipeline.run({
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question}
        })
        return result["llm"]["replies"][0].text 