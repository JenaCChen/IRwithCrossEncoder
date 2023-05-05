from elasticsearch_dsl import Document, Keyword, analyzer, Index, Text, DenseVector, Date, token_filter, Nested
from elasticsearch_dsl.connections import connections
from elasticsearch.helpers import bulk
import click

from load_data import load_nf_corpus


class ESDocument(Document):
	doc_id = Keyword()
	title = Text()
	content = Text(
		analyzer="english"
	)
	full_content = Text(
		analyzer="english"
	)
	domain = Text()
	annotation = Nested()
	sbert_embedding = DenseVector(
		dims=768
	)

	def save(self, *args, **kwargs):
		return super(ESDocument, self).save(*args, **kwargs)


class ESIndex(object):
	def __init__(
		self,
		index_name,
		docs
	):
		connections.create_connection(hosts=["localhost"], timeout=100, alias="default")
		self.index = index_name
		es_index = Index(self.index)

		if es_index.exists():
			es_index.delete()

		es_index.document(ESDocument)
		es_index.create()
		if docs is not None:
			self.load(docs)

	@staticmethod
	def _populate_doc(docs):
		for i, doc in enumerate(docs):
			es_doc = ESDocument(_id=i)
			es_doc.doc_id = doc["id"]
			es_doc.title = doc["title"]
			es_doc.content = doc["content"]
			es_doc.full_content = doc["full_content"]
			es_doc.domain = doc["domain"]
			es_doc.annotation = doc["annotation"]
			es_doc.sbert_embedding = doc["sbert_embedding"]
			yield es_doc

	def load(self, docs):
		bulk(
            connections.get_connection(),
            (
                d.to_dict(
                    include_meta=True, skip_empty=False
                )
                for d in self._populate_doc(docs)
            ),
        )


class IndexLoader:
	def __init__(self, index_name, docs):
		self.index_name = index_name
		self.docs = docs

	def load(self):
		ESIndex(self.index_name, self.docs)
		print("-----------loading finished-----------")

	@classmethod
	def from_folder(cls, index_name, nf_path):
		docs = load_nf_corpus(nf_path)
		return IndexLoader(index_name, docs)

@click.command()
@click.option("--index_name", default="es_corpus")
@click.option("--nf_path", default="nfcorpus")
def main(index_name, nf_path):
	index_loader = IndexLoader.from_folder(index_name, nf_path)
	index_loader.load()

if __name__ == "__main__":
    main()



