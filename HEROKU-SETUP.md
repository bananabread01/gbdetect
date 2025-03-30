# Heroku Deployment Guide

This guide provides step-by-step instructions for deploying the backend to Heroku.

## Setup Process

### 1. Create a Heroku Account

- Go to [Heroku](https://www.heroku.com/) and sign up for an account
- For free tier, you may need to have a GitHub Student Developer Pack, otherwise you'll need to provide credit card details for verification (but won't be charged for free tier)

### 2. Install Heroku CLI

Download and install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli):

- **Windows**: Download the installer
- **Mac**: `brew tap heroku/brew && brew install heroku`
- **Ubuntu**: `sudo snap install --classic heroku`

### 3. Login to Heroku

```bash
heroku login
```

### 4. Create a Heroku App

```bash
# Navigate to your repository root
cd path/to/your/repo

# Create a new Heroku app
heroku create your-app-name
```

### 5. Configure GitHub Actions for Deployment

1. Get your Heroku API key:
   ```bash
   heroku authorizations:create
   ```
   Or find it in your Heroku account settings.

2. Add the following secrets to your GitHub repository:
   - `HEROKU_API_KEY`: Your Heroku API key
   - `HEROKU_APP_NAME`: The name of your Heroku app (e.g., `your-app-name`)
   - `HEROKU_EMAIL`: Your Heroku account email
   - `BACKEND_URL`: Your Heroku app URL (e.g., https://your-app-name.herokuapp.com)

### 6. Update CORS Configuration

Edit the `backend/app.py` file to include your GitHub Pages URL in the CORS configuration:

```python
CORS(app, origins=[
    "http://localhost:5173",  # Local development
    "https://yourusername.github.io/gbdetect"  # GitHub Pages
])
```

### 7. Manual Deployment Option

If you prefer to deploy manually:

```bash
# Navigate to your backend directory
cd backend

# Login to Heroku
heroku login

# Create a Heroku app if you haven't already
heroku create your-app-name

# Initialize git if not already done
git init
git add .
git commit -m "Initial backend commit"

# Add Heroku remote
heroku git:remote -a your-app-name

# Push to Heroku
git push heroku master
```

### 8. Verify Deployment

Your API should be accessible at:
```
https://your-app-name.herokuapp.com/predict
```

## Configuration for Large Models

If your ML model is large, you may need to:

1. Use Git LFS for storing your model file:
   ```bash
   # Install Git LFS
   git lfs install
   
   # Track your model files
   git lfs track "*.pth"
   
   # Add, commit and push
   git add .gitattributes
   git add models/*.pth
   git commit -m "Add model files with Git LFS"
   git push
   ```

2. Or use an external storage service like AWS S3 and modify your code to download the model on startup.

## Troubleshooting

### Viewing Logs

```bash
heroku logs --tail
```

### Common Issues

1. **Build failures**: Check if all dependencies are properly specified in `requirements.txt`
2. **Memory issues**: Large ML models may exceed Heroku's free tier memory limits
3. **Timeout during startup**: Model loading may exceed Heroku's 60-second boot timeout
4. **Slug size limit**: Heroku has a 500MB limit for app size (code + dependencies)

### Note on Free Tier Limitations

- Free dynos sleep after 30 minutes of inactivity
- Limited to 550-1000 dyno hours per month
- Maximum of 5 free apps allowed 