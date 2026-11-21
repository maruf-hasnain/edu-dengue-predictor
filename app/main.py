from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model artifacts (e.g., artifacts/dengue_pipeline.joblib) will be loaded once here at startup
    print("Starting up Dengue Early-Warning Server...")
    yield
    print("Shutting down server...")


app = FastAPI(
    title="Urban Dengue Outbreak Early-Warning System",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
