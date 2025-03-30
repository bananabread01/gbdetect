# Gallbladder Cancer Detection

AI-powered Ultrasound Scan analysis for early Gallbladder Cancer detection.

## Project Structure

- **Frontend**: Vite React with TypeScript (in `/frontend` directory)
- **Backend**: Flask API with PyTorch for image analysis (in `/backend` directory)

## Deployment with GitHub Actions

This project is set up for automated deployment using GitHub Actions:

- Frontend is deployed to GitHub Pages
- Backend is deployed to Render

### Setup Instructions

1. **GitHub Repository Setup**:
   - Push this code to a GitHub repository
   - Enable GitHub Pages in repository settings: Settings > Pages > Build and deployment > Source > GitHub Actions

2. **Render Setup**:
   - Create a new Web Service on [Render](https://render.com/)
   - Connect your GitHub repository
   - Select the "Backend" directory
   - Set the environment to Python
   - Add the following environment variables:
     - `PYTHON_VERSION`: 3.10.0
     - `PATH_TO_MODEL`: models/MILprototype3.pth
   - Get your Render service ID and API key for GitHub secrets

3. **Update CORS Configuration**:
   - In `backend/app.py`, replace `"https://yourusername.github.io"` with your actual GitHub Pages URL (e.g., `"https://your-username.github.io/gbdetect"`)

4. **GitHub Secrets**:
   - Add the following secrets to your GitHub repository:
     - `RENDER_SERVICE_ID`: Your Render service ID
     - `RENDER_API_KEY`: Your Render API key
     - `BACKEND_URL`: Your Render backend URL (e.g., https://your-app.onrender.com)

5. **Environment Variables**:
   - The GitHub Actions workflow will automatically create `.env.production` file during build
   - For local development, copy `.env.example` to `.env.local` and update as needed

6. **Deploy**:
   - Push to main branch to trigger deployments
   - Alternatively, manually trigger workflows from GitHub Actions tab

## Local Development

1. **Frontend**:
   ```bash
   cd frontend
   yarn install
   # Copy example env file and update it
   cp .env.example .env.local
   # Edit .env.local with your local backend URL
   yarn dev
   ```

2. **Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   python app.py
   ```

## Model Information

The model used for gallbladder cancer detection is based on an Attention-MIL (Multiple Instance Learning) architecture, designed for processing ultrasound images. 