// API configuration

// Use environment variable for API URL or fallback to localhost for development
export const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

// Remove slashes from the API URL
const cleanApiUrl = API_URL.endsWith('/') ? API_URL.slice(0, -1) : API_URL;

// API endpoints
export const ENDPOINTS = {
  predict: `${cleanApiUrl}/predict`,
};

// Helper function for API requests
export async function uploadImage(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  console.log("Sending request to:", ENDPOINTS.predict);

  const response = await fetch(ENDPOINTS.predict, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Server error: Unable to process the image.");
  }

  return await response.json();
} 