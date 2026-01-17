import React, { useState } from "react";
import WebcamCapture from "../components/WebcamCapture";
import UploadCard from "../components/UploadCard";
import ResultCard from "../components/ResultCard";
import Loader from "../components/Loader";
import { verifyBiometrics } from "../services/api";

const Verify = () => {
  const [idImage, setIdImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);

  // Receives: { fps, rgb_frames, selfie_image }
  const handleVerify = async (verificationData) => {
    if (!idImage) {
      alert("Please upload a Static Shadow (ID) first.");
      return;
    }

    if (!verificationData?.selfie_image) {
      alert("Selfie capture failed. Please try again.");
      return;
    }

    setIsProcessing(true);
    setResult(null);

    try {
      const data = await verifyBiometrics(idImage, verificationData);
      setResult(data);
    } catch (error) {
      console.error("Verification Engine Error:", error);
      alert("Verification failed. Check backend logs.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white p-8">
      <div className="max-w-6xl mx-auto">
        <header className="mb-10 text-center">
          <h1 className="text-3xl font-bold tracking-tighter mb-2 uppercase">
            Identity Trial
          </h1>
          <p className="text-slate-500 font-mono text-sm tracking-widest uppercase">
            Initializing Biometric Sensors
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
          {/* Input Section */}
          <div className="space-y-8">
            <section>
              <h3 className="text-xs font-mono text-blue-500 mb-4 uppercase tracking-[0.2em]">
                Step 1: Static Shadow (ID)
              </h3>
              <UploadCard onUpload={setIdImage} selectedImage={idImage} />
            </section>

            <section>
              <h3 className="text-xs font-mono text-blue-500 mb-4 uppercase tracking-[0.2em]">
                Step 2: Living Form (Selfie)
              </h3>

              <WebcamCapture
                onVerify={handleVerify}
                isProcessing={isProcessing}
              />
            </section>
          </div>

          {/* Analysis Section */}
          <div className="bg-slate-900/40 rounded-3xl border border-white/5 p-8 relative min-h-[400px]">
            <h3 className="text-xs font-mono text-slate-500 mb-8 uppercase tracking-[0.2em]">
              Engine Logic Output
            </h3>

            {isProcessing ? (
              <Loader text="Analyzing biometric signals..." />
            ) : result ? (
              <ResultCard data={result} />
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-slate-600 opacity-50">
                <p className="text-sm font-mono tracking-widest">
                  AWAITING BIOMETRIC SIGNALS
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Verify;
