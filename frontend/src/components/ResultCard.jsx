const ResultCard = ({ data }) => {
  if (!data) return null;

  const {
    verdict,
    liveness,
    bpm,
    match_confidence
  } = data;

  const isVerified = verdict === "VERIFIED";

  return (
    <div className={`rounded-2xl p-6 border ${
      isVerified ? "border-green-500/40" : "border-red-500/40"
    }`}>
      <h2 className={`text-xl font-bold mb-6 ${
        isVerified ? "text-green-400" : "text-red-400"
      }`}>
        {isVerified ? "Verification Successful" : "Verification Failed"}
      </h2>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-slate-900/60 p-4 rounded-xl">
          <p className="text-xs text-slate-400 mb-1">Match Confidence</p>
          <p className="text-2xl font-bold">
            {match_confidence}%
          </p>
        </div>

        <div className="bg-slate-900/60 p-4 rounded-xl">
          <p className="text-xs text-slate-400 mb-1">Liveness Signal</p>
          <p className="text-lg font-semibold">
            {liveness ? "Living Human" : "Digital Ghost"}
          </p>
        </div>
      </div>

      <div className="text-sm text-slate-400">
        Heart Rate: <span className="text-white font-medium">{bpm} BPM</span>
      </div>
    </div>
  );
};

export default ResultCard;
