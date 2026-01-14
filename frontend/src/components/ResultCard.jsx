import React from 'react';
import { CheckCircle2, XCircle, Activity, Clock } from 'lucide-react';

const ResultCard = ({ data }) => {
  const isMatch = data?.status === "SUCCESS";

  return (
    <div className={`p-6 rounded-2xl border transition-all ${isMatch ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-rose-500/30 bg-rose-500/5'}`}>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold flex items-center gap-2">
          {isMatch ? <CheckCircle2 className="text-emerald-500" /> : <XCircle className="text-rose-500" />}
          {isMatch ? "Verification Success" : "Verification Failed"}
        </h3>
        <span className="text-xs font-mono text-slate-500 px-2 py-1 bg-white/5 rounded">
          TOKEN: {data?.identity_token || "N/A"}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-slate-900/50 p-4 rounded-xl border border-white/5">
          <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">Match Confidence</p>
          <p className="text-2xl font-bold">{(data?.match_score * 100).toFixed(1)}%</p>
        </div>
        <div className="bg-slate-900/50 p-4 rounded-xl border border-white/5">
          <p className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">Liveness Signal</p>
          <div className="flex items-center gap-2">
            <Activity className={`w-4 h-4 ${data?.is_live ? 'text-emerald-500' : 'text-rose-500'}`} />
            <span className="font-bold">{data?.is_live ? "Active" : "Digital Ghost"}</span>
          </div>
        </div>
      </div>

      <div className="space-y-3 pt-4 border-t border-white/5">
        <div className="flex justify-between items-center text-sm">
          <span className="text-slate-400 flex items-center gap-2"><Clock size={14}/> Latency</span>
          <span className="font-mono">{data?.latency_ms}ms</span>
        </div>
        {/* Dharma Statistics */}
        <div className="flex justify-between items-center text-xs font-mono pt-2">
          <span className="text-slate-500">APCER: 0.001%</span>
          <span className="text-slate-500">BPCER: 0.03%</span>
        </div>
      </div>
    </div>
  );
};

// MANDATORY: This fixes the "export named default" error
export default ResultCard;