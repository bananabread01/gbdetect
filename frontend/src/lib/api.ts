// API configuration

// Use environment variable for API URL or fallback to localhost for development
export const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

// API endpoints
export const ENDPOINTS = {
  predict: `${API_URL}/predict`,
};

// Helper function for API requests
export async function uploadImage(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(ENDPOINTS.predict, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Server error: Unable to process the image.");
  }

  return await response.json();
} 