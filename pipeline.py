from transformers import pipeline
import pandas as pd

# pipeline is a high-level interface for using pre-trained models (don't need to load the model manually/ worry about the details)
# Here we use the text-classification pipeline to classify the sentiment of a text

text = """Dear Amazon, last week I ordered an Optimus Prime action figure
from your online store in Germany. Unfortunately, when I opened the package,
I discovered to my horror that I had been sent an action figure of Megatron
instead! As a lifelong enemy of the Decepticons, I hope you can understand my
dilemma. To resolve the issue, I demand an exchange of Megatron for the
Optimus Prime figure I ordered. Enclosed are copies of my records concerning
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

classifier = pipeline("text-classification")
outputs = classifier(text)
pd.dataframe(outputs)
