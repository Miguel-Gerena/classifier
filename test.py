
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, GPT2Config
from transformers import LongformerForSequenceClassification, LongformerTokenizer, LongformerConfig
from transformers import PreTrainedTokenizerFast


tokenizer = DistilBertTokenizer.from_pretrained("classifier_weights/DistilBERT (Base-Uncased), G06F (Abstract) [Training_ 2011-2013]/dbert_G06F_train2011to13_tokenizer") 
model = DistilBertForSequenceClassification.from_pretrained("classifier_weights/DistilBERT (Base-Uncased), G06F (Abstract) [Training_ 2011-2013]/dbert_G06F_train2011to13")