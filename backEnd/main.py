from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router

app = FastAPI()

# Allow CORS from Chrome extension (replace with your extension ID)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://<your_extension_id>"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include API routes from the app module
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
