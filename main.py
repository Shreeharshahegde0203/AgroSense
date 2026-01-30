from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from typing import Optional

app = FastAPI(title="Irrigation AI Backend")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangFlow configuration (DataStax Cloud)
LANGFLOW_URL = "https://aws-us-east-2.langflow.datastax.com/lf/db1f6522-24fe-4ec9-8020-83f62c84c8f7/api/v1/run/7372d4d8-1866-4ffa-b18f-2f3dc6e2e239"
LANGFLOW_CHAT_URL = "https://aws-us-east-2.langflow.datastax.com/lf/db1f6522-24fe-4ec9-8020-83f62c84c8f7/api/v1/run/dc601587-83c3-433e-8e77-c6ed0fd224e6"
LANGFLOW_ORG_ID = "8bd5cc39-f56b-4028-a5f1-2144035a2f33"
LANGFLOW_TOKEN = "AstraCS:XUeFqiqrjjLlJZiPKGwOZcEu:c40137f7e499698b033eb8a7fba2b1d1b7fa21bbe491dcc923d9b646b94f5ba9"


class IrrigationInput(BaseModel):
    soil_moisture: float
    rainfall: float
    temperature: float
    evapotranspiration: float
    crop: str
    growth_stage: str
    season: str


class IrrigationResponse(BaseModel):
    recommendation: str
    status: str = "success"


class ChatInput(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    status: str = "success"


@app.get("/")
def root():
    return {"message": "Irrigation AI Backend", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(data: ChatInput):
    """
    Chat with agriculture assistant AI.
    """
    try:
        # Prepare request payload for LangFlow chat
        langflow_payload = {
            "input_value": data.message,
            "output_type": "chat",
            "input_type": "chat",
            "session_id": "agriculture-chat-session"
        }

        # Prepare headers for DataStax LangFlow
        headers = {
            "X-DataStax-Current-Org": LANGFLOW_ORG_ID,
            "Authorization": f"Bearer {LANGFLOW_TOKEN}",
            "Content-Type": "application/json"
        }

        print(f"Chat query: {data.message}")
        print(f"Chat URL: {LANGFLOW_CHAT_URL}")
        
        # Send request to LangFlow chat endpoint
        response = requests.post(
            f"{LANGFLOW_CHAT_URL}?stream=false",
            json=langflow_payload,
            headers=headers,
            timeout=60
        )

        print(f"Chat status: {response.status_code}")
        print(f"Chat response text: {response.text}")

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"LangFlow Chat API error: {response.status_code} - {response.text}"
            )

        # Parse response
        langflow_response = response.json()
        print(f"Chat response: {langflow_response}")

        # Extract reply
        try:
            reply = langflow_response["outputs"][0]["results"]["message"]["text"]
        except (KeyError, IndexError, TypeError):
            try:
                reply = langflow_response["outputs"][0]["outputs"][0]["results"]["message"]["text"]
            except (KeyError, IndexError, TypeError):
                if "result" in langflow_response:
                    reply = langflow_response["result"]
                else:
                    reply = "I'm having trouble responding right now. Please try again."

        return ChatResponse(reply=reply, status="success")

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Chat request timed out"
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to chat service"
        )
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Chat error: {str(e)}"
        )


@app.post("/analyze", response_model=IrrigationResponse)
async def analyze_irrigation(data: IrrigationInput):
    """
    Analyze irrigation requirements by sending input to LangFlow pipeline.
    """
    try:
        # Format input data as text block for LangFlow
        input_text = f"""Soil Moisture: {data.soil_moisture}
Rainfall: {data.rainfall}
Temperature: {data.temperature}
Evapotranspiration: {data.evapotranspiration}
Crop: {data.crop}
Growth Stage: {data.growth_stage}
Season: {data.season}"""

        # Prepare request payload for LangFlow
        langflow_payload = {
            "input_value": input_text,
            "output_type": "chat",
            "input_type": "chat",
            "session_id": "irrigation-session"
        }

        # Prepare headers for DataStax LangFlow
        headers = {
            "X-DataStax-Current-Org": LANGFLOW_ORG_ID,
            "Authorization": f"Bearer {LANGFLOW_TOKEN}",
            "Content-Type": "application/json"
        }

        # Send request to LangFlow (increased timeout for cloud)
        print(f"Sending to LangFlow: {input_text[:100]}...")
        response = requests.post(
            f"{LANGFLOW_URL}?stream=false",
            json=langflow_payload,
            headers=headers,
            timeout=120
        )
        print(f"LangFlow status: {response.status_code}")

        # Handle LangFlow errors
        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"LangFlow API error: {response.status_code}"
            )

        # Parse LangFlow response
        langflow_response = response.json()
        print(f"LangFlow response: {langflow_response}")
        
        # Extract the output text from LangFlow response
        # Try multiple possible response structures
        try:
            output = langflow_response["outputs"][0]["results"]["message"]["text"]
        except (KeyError, IndexError, TypeError):
            try:
                output = langflow_response["outputs"][0]["outputs"][0]["results"]["message"]["text"]
            except (KeyError, IndexError, TypeError):
                if "result" in langflow_response:
                    output = langflow_response["result"]
                else:
                    # Return the full response for debugging
                    output = f"Response structure: {str(langflow_response)}"

        return IrrigationResponse(
            recommendation=output,
            status="success"
        )

    except requests.exceptions.Timeout as e:
        print(f"Timeout error: {e}")
        raise HTTPException(
            status_code=504,
            detail="LangFlow request timed out after 120 seconds. Your flow might be taking too long to process."
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to LangFlow. Ensure it's running at http://localhost:7860"
        )
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected LangFlow response structure: {str(e)}"
        )
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
