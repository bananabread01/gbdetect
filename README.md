# Gallbladder Cancer Detection

AI-powered Ultrasound Scan analysis for early Gallbladder Cancer detection.

## Project Structure

- **Frontend**: Vite React with TypeScript (in `/frontend` directory)
- **Backend**: Flask API with PyTorch for image analysis (in `/backend` directory)

## Deployment with GitHub Actions

This project is set up for automated deployment using GitHub Actions:
- Frontend is deployed to GitHub Pages
- Backend is deployed to Heroku



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
