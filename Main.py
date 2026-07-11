import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="GLM 5.2 Precise Code Corrector")

# Initialize the GLM 5.2 Client
client = OpenAI(
    base_url=os.environ.get("GLM_BASE_URL", "https://api.z.ai/v1"),
    api_key=os.environ.get("GLM_API_KEY")
)

class CodePayload(BaseModel):
    code: str
    language: str

@app.get("/", response_class=HTMLResponse)
def home_interface():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GLM 5.2 Code Repair</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0e0e10; color: #e1e1e6; padding: 20px; margin: 0; }
            .container { max-width: 600px; margin: 0 auto; background: #18181c; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
            h2 { color: #00db75; margin-top: 0; font-size: 22px; text-align: center; }
            textarea { width: 100%; height: 250px; background: #222226; color: #fff; border: 1px solid #2e2e34; border-radius: 8px; padding: 12px; box-sizing: border-box; font-family: monospace; font-size: 14px; }
            select { width: 100%; padding: 14px; background: #222226; color: #fff; border: 1px solid #2e2e34; border-radius: 8px; margin: 15px 0; font-size: 16px; }
            button { width: 100%; padding: 16px; border: none; border-radius: 8px; font-weight: bold; font-size: 16px; cursor: pointer; transition: 0.2s; }
            .btn-start { background: #00db75; color: #000; }
            .btn-start:disabled { background: #2e2e34; color: #727278; cursor: not-allowed; }
            .btn-stop { background: #ffffff; color: #000; display: none; margin-top: 10px; }
            .status { margin-top: 15px; font-weight: bold; text-align: center; color: #a8a8b3; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>GLM 5.2 Engine: Direct Code Repair</h2>
            
            <textarea id="codeInput" placeholder="// Paste your broken or unoptimized code here..."></textarea>
            
            <select id="langInput">
                <option value="python">Python</option>
                <option value="javascript">JavaScript / TypeScript</option>
                <option value="golang">Go (Golang)</option>
                <option value="rust">Rust</option>
                <option value="java">Java / C++</option>
            </select>
            
            <button class="btn-start" id="startBtn" onclick="startSession()">Run Deep Code Fix</button>
            <button class="btn-stop" id="stopBtn" onclick="stopAndDownload()">Download Fixed Code File</button>
            
            <div class="status" id="statusMessage">Awaiting code input...</div>
        </div>

        <script>
            let perfectCodeOutput = "";
            let selectedLang = "";

            async function startSession() {
                const codeData = document.getElementById("codeInput").value.trim();
                selectedLang = document.getElementById("langInput").value;
                
                if (!codeData) {
                    alert("Please paste your code first!");
                    return;
                }

                const startBtn = document.getElementById("startBtn");
                const statusMsg = document.getElementById("statusMessage");

                startBtn.disabled = true;
                statusMsg.style.color = "#00db75";
                statusMsg.innerText = "GLM 5.2 is reasoning and compiling the optimal fix...";

                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ code: codeData, language: selectedLang })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        perfectCodeOutput = result.fixed_code;
                        statusMsg.innerText = "Perfect code generated successfully!";
                        document.getElementById("stopBtn").style.display = "block";
                    } else {
                        throw new Error(result.error || "Processing failed.");
                    }
                } catch (error) {
                    statusMsg.style.color = "#ff4a4a";
                    statusMsg.innerText = "Error: " + error.message;
                    startBtn.disabled = false;
                }
            }

            function stopAndDownload() {
                if (!perfectCodeOutput) return;

                // Extension mapping for direct file opening on mobile
                const extMap = { python: "py", javascript: "js", golang: "go", rust: "rs", java: "java" };
                const fileExt = extMap[selectedLang] || "txt";

                const blob = new Blob([perfectCodeOutput], { type: "text/plain;charset=utf-8" });
                const url = URL.createObjectURL(blob);
                
                const downloadLink = document.createElement("a");
                downloadLink.href = url;
                downloadLink.download = `fixed_output.${fileExt}`;
                
                document.body.appendChild(downloadLink);
                downloadLink.click();
                
                document.body.removeChild(downloadLink);
                URL.revokeObjectURL(url);

                // Reset page state
                perfectCodeOutput = "";
                document.getElementById("startBtn").disabled = false;
                document.getElementById("stopBtn").style.display = "none";
                document.getElementById("codeInput").value = "";
                document.getElementById("statusMessage").innerText = "File downloaded to Chrome folder. Ready for next file.";
            }
        </script>
    </body>
    </html>
    """

@app.post("/analyze")
async def analyze_code(payload: CodePayload):
    try:
        # Request strict, pure-code response using maximum reasoning paths
        response = client.chat.completions.create(
            model="glm-5.2",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are an absolute expert code optimization and repair machine. "
                        "The user wants the final, highly accurate, correct code solution. "
                        "Do not include comments, text explanations, bullet points, markdown code blocks, or chat filler. "
                        "Output ONLY the final, clean, working production code itself. Nothing else."
                    )
                },
                {
                    "role": "user", 
                    "content": payload.code
                }
            ],
            extra_body={"reasoning_effort": "max"}
        )
        
        # Strip out any unexpected markdown block wrapping if the LLM adds it anyway
        raw_output = response.choices[0].message.content.strip()
        if raw_output.startswith("```"):
            lines = raw_output.splitlines()
            if lines[-1].startswith("```"):
                raw_output = "\n".join(lines[1:-1]).strip()

        return {"success": True, "fixed_code": raw_output}
        
    except Exception as e:
        return {"success": False, "error": str(e)}
