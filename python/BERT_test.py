from bert_serving.client import BertClient
bc = BertClient()
print(bc.encode(['First do it', 'then do it right', 'then do it better']))