import json
import pytest
from sentence_transformers import SentenceTransformer, util
from neww import InternshipChatbot  # Import the InternshipChatbot class

def load_test_cases():
    with open('questions_answers.json', 'r') as f:
        return json.load(f)

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Initialize the chatbot instance once before all tests
chatbot = InternshipChatbot(embeddings_path="embeddings")  # Ensure correct path to embeddings

@pytest.mark.parametrize("test_case", load_test_cases())
def test_chatbot_responses(test_case):
    question = test_case['question']
    expected_sentence = test_case['expected_answer']

    # Access the chatbot response
    actual_response = chatbot.get_response(question)['response']  # Access the 'response' field

    # Generate embeddings for expected and actual responses
    try:
        embedding1 = model.encode(expected_sentence, convert_to_tensor=True)
        embedding2 = model.encode(actual_response, convert_to_tensor=True)
    except Exception as e:
        pytest.fail(f"Error generating embeddings: {e}")

    # Calculate cosine similarity
    cosine_scores = util.cos_sim(embedding1, embedding2)
    similarity_score = cosine_scores.item()

    similarity_threshold = 0.7  # Adjust this threshold as needed

    # Check for match criteria based on expected response and enable for 'exact' type tests
    if test_case.get('type') == 'exact':
        assert actual_response == expected_sentence, (
            f"Failed for question: '{question}'.\n"
            f"Expected: '{expected_sentence}',\n"
            f"Got: '{actual_response}'."
        )
    else:
        assert similarity_score >= similarity_threshold, (
            f"Failed for question: '{question}'.\n"
            f"Expected: '{expected_sentence}',\n"
            f"Got: '{actual_response}'.\n"
            f"Similarity score: {similarity_score}"
        )

    print(f"PASSED: '{question}'")

if __name__ == '__main__':
    pytest.main()