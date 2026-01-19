import React, { useState } from "react";
import UploadCard from "../components/UploadCard";
import WebcamCapture from "../components/WebcamCapture";
import ResultCard from "../components/ResultCard";
import Loader from "../components/Loader";
import { verifyFace } from "../services/api";

const Verify = () => {
  const [idImage, setIdImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [bbox, setBbox] = useState(null);
  const [imgSize, setImgSize] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = (file) => {
    setIdImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setBbox(null);
    setResult(null);
    setImgSize(null);
  };

  const handleVerify = async ({ selfieBlob }) => {
    if (!idImage) {
      alert("Upload ID image first");
      return;
    }

    setLoading(true);
    try {
      const data = await verifyFace(idImage, selfieBlob);
      setBbox(data.bbox);
      setResult(data);
    } catch (e) {
      alert("Verification failed");
    } finally {
      setLoading(false);
    }
  };

  // ---- Scale bbox correctly to displayed image ----
  const scaledBox =
    bbox && imgSize
      ? {
          left: `${(bbox.x / bbox.img_w) * imgSize.width}px`,
          top: `${(bbox.y / bbox.img_h) * imgSize.height}px`,
          width: `${(bbox.w / bbox.img_w) * imgSize.width}px`,
          height: `${(bbox.h / bbox.img_h) * imgSize.height}px`,
        }
      : null;

  return (
    <div className="min-h-screen bg-slate-950 text-white px-6 py-10">
      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-10 items-start">

        {/* LEFT PANEL */}
        <div className="space-y-6">
          <UploadCard onUpload={handleUpload} selectedImage={idImage} />

          {previewUrl && (
            <div className="relative w-full max-w-sm mx-auto rounded-xl overflow-hidden border border-white/10 bg-black">
              
              {/* IMAGE */}
              <img
                src={previewUrl}
                alt="ID Preview"
                className="w-full h-64 object-contain bg-black relative z-10"
                onLoad={(e) =>
                  setImgSize({
                    width: e.target.clientWidth,
                    height: e.target.clientHeight,
                  })
                }
              />

              {/* GREEN BOX (FINAL FIX) */}
              {scaledBox && (
                <div
                  className="absolute border-2 border-green-500 pointer-events-none z-20"
                  style={scaledBox}
                />
              )}
            </div>
          )}

          <WebcamCapture onVerify={handleVerify} />
        </div>

        {/* RIGHT PANEL */}
        <div className="min-h-[300px] bg-slate-900/40 rounded-2xl border border-white/5 p-6">
          {loading ? <Loader /> : <ResultCard data={result} />}
        </div>

      </div>
    </div>
  );
};

export default Verify;
