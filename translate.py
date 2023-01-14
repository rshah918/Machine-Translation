import torch
from model import Seq2SeqTransformer
from LanguageTranslation import translate
'''
Top level code to translate english sentences to spanish
'''
#deploy to apple silicon
DEVICE="mps" if torch.backends.mps.is_available() else "cpu"
#load model
neural_net = Seq2SeqTransformer().to(DEVICE)
neural_net.load_state_dict(torch.load("trained_model", map_location=torch.device('mps')))
#translate!
while True:
    inp = input("Enter an English Sentence >")

    print("Spanish Translation: ", translate(neural_net, inp))