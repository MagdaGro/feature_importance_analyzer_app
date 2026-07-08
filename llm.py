import json
from openai import OpenAI
from prompts import SYSTEM_PROMPT
from typing import List
from schemas import BusinessReport



def build_llm_payload(
                    target_col,
                    problem_type,
                    best_model,
                    best_metrics,
                    fi_dict,
                    fi_df
                ):
                """Builds the payload for the LLM report generation."""
                return {
                    "target_variable": target_col,
                    "problem_type": problem_type,
                    "model_name": type(best_model).__name__,
                    "model_metrics": best_metrics,
                    "importance_method": fi_dict["method"],
                    "top_features": fi_df.head(10).to_dict("records"
                    )
                }


def generate_report(payload, api_key, max_retries=2):
                """Generates a business report using the LLM based on the provided payload. """
                client = OpenAI(api_key=api_key)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, default=str)}           
                    ]

                for attempt in range(max_retries + 1):  
                    
                    response = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=messages,
                        temperature=0.2,
                        )
                    raw_output = response.choices[0].message.content 

                    try: 

                        parsed = BusinessReport.model_validate_json(raw_output)   
                        return parsed

                    except Exception as e:

                        messages.append({"role": "assistant", "content": raw_output})
                        messages.append({"role": "user",
                    "content": f"""
            Your previous output was not valid JSON.

            Error:
            {str(e)}

            Please return valid JSON only.
            """
            })
                        if attempt == max_retries:

                            raise ValueError(f"LLM failed after {attempt + 1} attempts.")
                        
def validate_report(report: BusinessReport):
                """Validates the generated business report to ensure it meets the required criteria."""

                assert len(report.key_drivers) > 0, "At least one key driver must be identified."
                assert len(report.recommendations) > 0, "Too few recommendations."
                assert report.executive_summary is not None, "Executive summary is required."
                
                for f in report.key_drivers:
                    assert f.importance >= 0, f"Feature importance must be non-negative: {f.feature}"

                return True

