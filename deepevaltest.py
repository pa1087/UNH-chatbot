import os
import requests
import json
from deepeval.metrics import AnswerRelevancyMetric, GEval, FaithfulnessMetric
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test

# Set the OpenAI API key as an environment variable
load_dotenv()
#API_KEY = "your api key"  # Replace with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = API_KEY

# Load the test cases from a JSON file
def retrieve_test_cases(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        return data["test_cases"]

# Get chatbot response from the local server
def get_chatbot_response(question):
    """Fetch a response from the chatbot server for a given question."""
    url = 'http://127.0.0.1:5000/ask'
    payload = {"message": question}

    try:
        # Send POST request and handle the response
        with requests.post(url, json=payload) as response:
            response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
            return response.json().get("response", "")
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err} for question: {question}"
    except requests.exceptions.RequestException as req_err:
        return f"Request exception occurred: {req_err} for question: {question}"
    except json.JSONDecodeError:
        return f"Response returned is not JSON formatted for the question: {question}"

    return None

def metric_evaluation(test_case, relevancy_threshold=0.7, faithfulness_threshold=0.5):
    """Configurable metric evaluation with flexible thresholds."""
    relevancy_metric = AnswerRelevancyMetric(threshold=relevancy_threshold)
    faithfulness_metric = FaithfulnessMetric(threshold=faithfulness_threshold)

    actual_output = get_chatbot_response(test_case["question"])
    llm_test_case = LLMTestCase(
        input=test_case["question"],
        actual_output=actual_output,
        expected_output=test_case["expected_answer"],
        retrieval_context=test_case.get("retrieval_context")
    )

    relevancy_metric.measure(llm_test_case)
    faithfulness_metric.measure(llm_test_case)

    return {
        "metrics": {
            "relevancy": {
                "score": relevancy_metric.score or 0,
                "threshold": relevancy_threshold
            },
            "faithfulness": {
                "score": faithfulness_metric.score or 0,
                "threshold": faithfulness_threshold
            }
        },
        "test_details": {
            "question": test_case["question"],
            "actual_answer": actual_output
        }
    }


def main():
    """Main function to execute test case evaluations."""
    test_cases = retrieve_test_cases('deqna.json')

    passed_count = 0
    failed_count = 0

    with open('metric_results.txt', 'w') as result_file:
        for case in test_cases:
            result = metric_evaluation(case)

            # Determine if the test case passes or fails
            relevancy_passed = result['metrics']['relevancy']['score'] >= result['metrics']['relevancy']['threshold']
            faithfulness_passed = result['metrics']['faithfulness']['score'] >= result['metrics']['faithfulness'][
                'threshold']
            test_passed = relevancy_passed and faithfulness_passed

            if test_passed:
                passed_count += 1
                status = "Passed"
            else:
                failed_count += 1
                status = "Failed"

            # Updated to match new result structure
            result_output = (
                f"Question: {result['test_details']['question']}\n"
                f"Actual Answer: {result['test_details']['actual_answer']}\n"
                f"Relevancy Score: {result['metrics']['relevancy']['score']}\n"
                f"Relevancy Threshold: {result['metrics']['relevancy']['threshold']}\n"
                f"Faithfulness Score: {result['metrics']['faithfulness']['score']}\n"
                f"Faithfulness Threshold: {result['metrics']['faithfulness']['threshold']}\n"
                f"Status: {status}\n"
                "------------------------------------------\n"
            )

            result_file.write(result_output)

    # Write the summary of passed and failed counts
    summary_output = (
        f"Total Test Cases: {len(test_cases)}\n"
        f"Passed: {passed_count}\n"
        f"Failed: {failed_count}\n"
    )

    with open('metric_results.txt', 'a') as result_file:
        result_file.write(summary_output)


if __name__ == "__main__":
    main()