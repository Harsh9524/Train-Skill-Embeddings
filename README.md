# Train-Skill-Embeddings
Practical Approach to Word Embeddings using Word2Vec and Gensim.
Word Embeddings are used to make a relation between a bag of words which are related to each other in some sense. For instance if there is a dataset of 100 candidates containing skills of each candidate in the form of a list, using word embeddings we can make a relation between each word so that Java and C++ are more related than Java and Business Development.

Requirements

- Genism (pip install genism)
- tensorflow (pip install tensorflow==1.*)
- python3.6

Usage
1. Download the dataset from https://drive.google.com/file/d/1JlGepY8wezSuoiAP1tCi-3TIt-DWGXOe/view
2. Put the 'all_linkedin_skill_data' file in the same folder as main.py
3. run the main.py file using 'python main.py' command.
4. for visualizing with TensorBoard use 'tensorboard --logdir=projections --port=8000' command.
5. Open 'http://localhost:8000/' in any browser.

If you are facing any difficulties while cloning or running the .py file so please raise a issue or read the medium article mentioned above.
