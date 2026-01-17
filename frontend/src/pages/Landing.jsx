import React from "react";
import { motion } from "framer-motion";
import { 
  Zap, Lock, Cpu, Globe, Eye, RefreshCcw, 
  Shield, Database, Github, Twitter, Linkedin, 
  Command, ChevronRight, ScanFace, Activity, Hexagon
} from "lucide-react";

export default function Landing() {
  return (
    <div className="min-h-screen bg-[#010409] text-slate-200 font-sans selection:bg-cyan-500/30">
      
      {/* --- HERO SECTION --- */}
      <section className="relative pt-24 pb-12 px-6 overflow-hidden">
        {/* Animated Background Mesh */}
        <div className="absolute inset-0 z-0 opacity-[0.05]" 
             style={{ backgroundImage: `radial-gradient(circle at 2px 2px, #22d3ee 1px, transparent 0)`, backgroundSize: '40px 40px' }} />
        
        <div className="max-w-7xl mx-auto grid lg:grid-cols-5 gap-12 items-center relative z-10">
          
          <motion.div 
            className="lg:col-span-3"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-[10px] font-mono mb-6 uppercase tracking-[0.3em]">
              <Hexagon size={12} className="animate-spin-slow" />
              Neural_Link: Established // Node_01
            </div>
            
            <h1 className="text-6xl md:text-[100px] font-black leading-[0.8] tracking-tighter text-white mb-8 uppercase italic">
              NEURAL<br />
              <motion.span 
                animate={{ opacity: [1, 0.5, 1] }}
                transition={{ duration: 4, repeat: Infinity }}
                className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-cyan-600"
              >
                LATTICE
              </motion.span>
            </h1>
            
            <p className="text-slate-400 text-lg max-w-lg leading-relaxed mb-10 border-l-2 border-cyan-500/50 pl-6 font-light">
              Autonomous identity orchestration. We replace static 2D verification with 
              <span className="text-white italic"> volumetric sub-dermal mesh</span> analysis.
            </p>

            <div className="flex gap-6">
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="group relative bg-cyan-500 text-slate-950 font-black px-10 py-5 flex items-center gap-3 tracking-tighter uppercase text-sm shadow-[0_0_30px_rgba(6,182,212,0.4)]"
              >
                Initialize Scan <ChevronRight size={18} />
              </motion.button>
            </div>
          </motion.div>

          {/* THE 3D FACE: MAXIMUM MOTION */}
          <div className="lg:col-span-2 relative flex justify-center items-center h-[500px]">
            <motion.div 
              animate={{ 
                rotateY: [0, 360],
                y: [0, -20, 0]
              }}
              transition={{ 
                rotateY: { duration: 20, repeat: Infinity, ease: "linear" },
                y: { duration: 5, repeat: Infinity, ease: "easeInOut" }
              }}
              className="relative preserve-3d"
            >
              {/* Spinning HUD Rings */}
              {[1, 2].map((i) => (
                <motion.div
                  key={i}
                  animate={{ rotate: i % 2 === 0 ? 360 : -360 }}
                  transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
                  className="absolute inset-[-40px] border border-cyan-500/10 rounded-full border-dashed"
                />
              ))}

              <div className="relative p-10 bg-slate-900/40 rounded-full border border-cyan-500/20 backdrop-blur-3xl shadow-[0_0_60px_rgba(34,211,238,0.15)]">
                <ScanFace size={240} strokeWidth={0.5} className="text-cyan-400 drop-shadow-[0_0_20px_cyan]" />
                
                {/* Horizontal Pulse Scan */}
                <motion.div 
                  animate={{ top: ["0%", "100%", "0%"] }}
                  transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                  className="absolute left-0 right-0 h-[2px] bg-cyan-400 shadow-[0_0_15px_cyan] z-20"
                />
              </div>

              {/* Data Node Tags */}
              <FloatingData x="120%" y="-30%" label="V_MESH: ACTIVE" />
              <FloatingData x="-120%" y="40%" label="LIVENESS: SYNC" />
            </motion.div>
          </div>
        </div>
      </section>

      {/* --- VISUAL TELEMETRY GRID (No Numbers) --- */}
      <section className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <VisualWidget label="Latency Buffer" icon={<Zap/>} status="Minimal" variant="wave" />
          <VisualWidget label="Network Load" icon={<Globe/>} status="Stable" variant="bars" />
          <VisualWidget label="Encryption" icon={<Lock/>} status="Quantum" variant="progress" />
          <VisualWidget label="Engine Sync" icon={<RefreshCcw/>} status="Syncing" variant="pulse" />
        </div>
      </section>

      {/* --- INDUSTRIAL FOOTER --- */}
      <footer className="bg-[#020617] border-t border-white/5 pt-20 pb-10 px-10 mt-20">
        <div className="max-w-7xl mx-auto grid md:grid-cols-4 gap-16 mb-16">
          <div className="col-span-2">
            <h2 className="text-2xl font-black text-white italic mb-6">Neural<span className="text-cyan-500 font-black">Lattice</span></h2>
            <p className="text-slate-500 text-sm max-w-sm leading-relaxed mb-8">
              Decentralized biometric infrastructure utilizing volumetric vision transformers for absolute human-presence proof.
            </p>
            <div className="flex gap-4">
              <FooterIcon icon={<Github size={18}/>} />
              <FooterIcon icon={<Twitter size={18}/>} />
              <FooterIcon icon={<Linkedin size={18}/>} />
            </div>
          </div>
          
          <div>
            <h4 className="text-[10px] font-mono text-cyan-500 uppercase tracking-widest mb-6">Navigation</h4>
            <ul className="space-y-4 text-xs font-mono text-slate-500 uppercase tracking-tighter">
              <li className="hover:text-white cursor-pointer transition-colors">Documentation</li>
              <li className="hover:text-white cursor-pointer transition-colors">Research_Logs</li>
              <li className="hover:text-white cursor-pointer transition-colors">API_Reference</li>
            </ul>
          </div>

          <div>
            <h4 className="text-[10px] font-mono text-cyan-500 uppercase tracking-widest mb-6">Sys_Log</h4>
            <div className="space-y-2 text-[9px] font-mono text-slate-600">
              <p>● UPTIME: 99.998%</p>
              <p>● ENCRYPTION: LATTICE_X2</p>
              <p>● STATUS: NOMINAL</p>
            </div>
          </div>
        </div>
        
        <div className="max-w-7xl mx-auto pt-10 border-t border-white/5 flex justify-between items-center text-[9px] font-mono text-slate-700 uppercase tracking-[0.4em]">
          <span>© 2026 NeuralLattice // Auth_Locked</span>
          <div className="flex gap-8">
            <span className="hover:text-white cursor-pointer">Privacy</span>
            <span className="hover:text-white cursor-pointer">Protocols</span>
          </div>
        </div>
      </footer>

      <style jsx>{`
        .preserve-3d { transform-style: preserve-3d; perspective: 1000px; }
        .animate-spin-slow { animation: spin 8s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}

/* --- TELEMETRY COMPONENTS --- */

function VisualWidget({ label, icon, status, variant }) {
  return (
    <motion.div 
      whileHover={{ scale: 1.02, backgroundColor: "rgba(255,255,255,0.02)" }}
      className="p-6 border border-white/5 bg-white/[0.01] transition-all group"
    >
      <div className="flex justify-between items-start mb-6">
        <div className="text-cyan-500 group-hover:scale-110 transition-transform">{icon}</div>
        <div className="text-[8px] font-mono text-cyan-400 border border-cyan-500/30 px-1.5 py-0.5 uppercase">{status}</div>
      </div>
      <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-4">{label}</p>
      
      {/* Visual instead of Numbers */}
      <div className="h-6 flex items-end gap-1">
        {variant === 'wave' && [1,2,3,4,5,6,7].map(i => (
          <motion.div 
            key={i}
            animate={{ height: [4, 20, 4] }}
            transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.1 }}
            className="w-1 bg-cyan-500/40"
          />
        ))}
        {variant === 'bars' && [1,2,3,4,5].map(i => (
          <div key={i} className={`w-full h-${i*2} bg-cyan-500/${i*20}`} />
        ))}
        {variant === 'progress' && (
          <div className="w-full h-1 bg-white/5 mb-2 relative">
            <motion.div 
              animate={{ width: ["20%", "90%", "60%", "95%"] }}
              transition={{ duration: 5, repeat: Infinity }}
              className="h-full bg-cyan-500 shadow-[0_0_10px_cyan]"
            />
          </div>
        )}
        {variant === 'pulse' && (
          <motion.div 
            animate={{ opacity: [0.2, 1, 0.2] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-full h-1 bg-cyan-500/50 shadow-[0_0_15px_cyan]"
          />
        )}
      </div>
    </motion.div>
  );
}

function FloatingData({ x, y, label }) {
  return (
    <motion.div 
      animate={{ y: ["-10%", "10%"] }} 
      transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
      className="absolute z-50 whitespace-nowrap"
      style={{ top: '50%', left: '50%', transform: `translate(${x}, ${y})` }}
    >
      <div className="flex items-center gap-2 bg-black/80 border border-cyan-500/30 px-3 py-1.5">
        <div className="w-1 h-1 bg-cyan-400 rotate-45" />
        <span className="text-[9px] font-mono text-cyan-400 tracking-tighter uppercase">{label}</span>
      </div>
    </motion.div>
  );
}

function FooterIcon({ icon }) {
  return (
    <div className="w-10 h-10 flex items-center justify-center border border-white/5 hover:border-cyan-500 transition-all text-slate-500 hover:text-cyan-400 cursor-pointer">
      {icon}
    </div>
  );
}