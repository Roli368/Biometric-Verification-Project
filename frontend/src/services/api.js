import axios from 'axios';

// The URL where your FastAPI backend is running
const API_BASE_URL = 'http://localhost:8000/api/v1';

export const verifyBiometrics = async (idFile, selfieBase64) => {
  const formData = new FormData();
  
  // Append the ID file (Static Shadow)
  formData.append('id_card', idFile);
  
  // Append the Webcam capture (Living Form)
  formData.append('selfie', selfieBase64);

  try {
    const response = await axios.post(`${API_BASE_URL}/verify`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error("API Error:", error.response?.data || error.message);
    throw error;
  }
};