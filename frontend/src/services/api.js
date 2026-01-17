import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

/**
 * @param {File} idFile        - uploaded ID image
 * @param {Object} livenessData
 * @param {string} livenessData.selfie_image - base64 webcam screenshot
 * @param {number} livenessData.fps
 * @param {Array}  livenessData.rgb_frames
 */
export const verifyBiometrics = async (idFile, livenessData) => {
  const formData = new FormData();

  // ---- ID CARD ----
  formData.append("id_card", idFile);

  // ---- SELFIE (convert base64 -> Blob) ----
  const base64ToBlob = (base64) => {
    const byteString = atob(base64.split(",")[1]);
    const mimeString = base64.split(",")[0].split(":")[1].split(";")[0];

    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);

    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ab], { type: mimeString });
  };

  const selfieBlob = base64ToBlob(livenessData.selfie_image);
  formData.append("selfie", selfieBlob, "selfie.jpg");

  // ---- LIVENESS DATA ----
  formData.append(
    "liveness",
    JSON.stringify({
      fps: livenessData.fps,
      rgb_frames: livenessData.rgb_frames,
    })
  );

  // ---- API CALL ----
  const response = await axios.post(
    `${API_BASE_URL}/verify`,
    formData
  );

  return response.data;
};
