import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

export const verifyFace = async (idFile, selfieBlob) => {
  const formData = new FormData();
  formData.append("id_image", idFile);
  formData.append("selfie_image", selfieBlob, "selfie.jpg");

  const res = await axios.post(
    `${API_BASE_URL}/verify-face`,
    formData
  );

  return res.data;
};

