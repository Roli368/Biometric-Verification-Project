import React from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { 
  Shield, Zap, Activity, ArrowRight, ScanFace, 
  Fingerprint, Cpu, Lock, Globe, Database 
} from "lucide-react";

export default function LandingPage() {
  const navigate = useNavigate();
  const { scrollY } = useScroll();
  
  // Parallax effects for the hero visual
  const y1 = useTransform(scrollY, [0, 500], [0, 200]);
  const opacity = useTransform(scrollY, [0, 300], [1, 0]);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.2 }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1 }
  };

  return (
    <div className="min-h-screen bg-[#030712] text-white font-sans selection:bg-cyan-500/30">
      {/* Dynamic Background Noise/Grid */}
      <div className="fixed inset-0 z-0 opacity-[0.03] pointer-events-none bg-[url('https://grainy-gradients.vercel.app/noise.svg')]"></div>
      <div className="fixed inset-0 z-0 bg-[linear-gradient(to_right,#1f2937_1px,transparent_1px),linear-gradient(to_bottom,#1f2937_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] opacity-20"></div>

      {/* Hero Section */}
      <section className="relative pt-20 pb-32 px-6 md:px-10 overflow-hidden">
        <div className="max-w-7xl mx-auto grid md:grid-cols-2 gap-16 items-center">
          
          <motion.div 
            initial="hidden"
            animate="visible"
            variants={containerVariants}
            className="relative z-10"
          >
            <motion.div variants={itemVariants} className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-[10px] font-mono mb-8 uppercase tracking-[0.3em]">
              <Cpu size={14} className="animate-spin-slow" />
              Quantum-Resistant Layer Active
            </motion.div>
            
            <motion.h1 variants={itemVariants} className="text-6xl md:text-8xl font-black leading-[0.85] tracking-tighter mb-8">
              IDENTITY <br /> 
              <span className="text-transparent bg-clip-text bg-gradient-to-b from-white to-slate-500">WITHOUT</span> <br />
              DOUBT.
            </motion.h1>

            <motion.p variants={itemVariants} className="text-slate-400 max-w-md text-xl font-light leading-relaxed mb-10">
              Reconciling the <span className="text-white italic">Static Shadow</span> with the <span className="text-white italic">Living Form</span> using sub-500ms Siamese Transformers.
            </motion.p>

            <motion.div variants={itemVariants} className="flex flex-wrap gap-4">
              <button 
                onClick={() => navigate('/verify')}
                className="group bg-white text-slate-950 font-bold px-8 py-4 rounded-full flex items-center gap-3 transition-all hover:bg-cyan-400 active:scale-95"
              >
                INITIATE TRIAL <ArrowRight size={20} />
              </button>
              <button className="px-8 py-4 rounded-full border border-white/10 font-bold hover:bg-white/5 transition-all">
                WHITEPAPER
              </button>
            </motion.div>
          </motion.div>

          {/* Interactive Hero Visual */}
          <motion.div style={{ y: y1, opacity }} className="relative flex justify-center items-center">
            <div className="absolute w-[120%] h-[120%] bg-cyan-500/5 blur-[120px] rounded-full" />
            <div className="relative w-80 h-80 md:w-96 md:h-96">
               <motion.div 
                 animate={{ rotate: 360 }}
                 transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
                 className="absolute inset-0 rounded-full border border-cyan-500/10 border-t-cyan-500/40"
               />
               <div className="absolute inset-4 rounded-full bg-slate-900/40 backdrop-blur-3xl border border-white/5 flex items-center justify-center overflow-hidden">
                  <ScanFace size={160} className="text-cyan-400/20" />
                  <motion.div 
                    animate={{ top: ["-10%", "110%"] }}
                    transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                    className="absolute left-0 right-0 h-[2px] bg-cyan-400 shadow-[0_0_20px_#22d3ee] z-20"
                  />
                  <div className="absolute inset-0 bg-gradient-to-b from-transparent via-cyan-500/5 to-transparent opacity-50" />
               </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Bento Grid */}
      <section id="stats" className="px-6 md:px-10 py-24 max-w-7xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard title="Accuracy" value="99.5%" desc="TAR @ FAR 1e-4" colSpan="md:col-span-2" />
          <StatCard title="Liveness" value="99.9%" desc="Anti-Spoofing" />
          <StatCard title="Latency" value="<500ms" desc="Edge Speed" />
        </div>
      </section>

      {/* Technical Pillars */}
      <section className="px-6 md:px-10 py-32 max-w-7xl mx-auto">
        <motion.div 
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="grid md:grid-cols-3 gap-12"
        >
          <div className="space-y-4">
            <Fingerprint className="text-cyan-400" size={40} />
            <h3 className="text-2xl font-bold">Swarupa Matching</h3>
            <p className="text-slate-400 leading-relaxed">Cross-domain reconciliation between grainy ID captures and high-fidelity live biometric streams.</p>
          </div>
          <div className="space-y-4">
            <Activity className="text-cyan-400" size={40} />
            <h3 className="text-2xl font-bold">Prana Detection</h3>
            <p className="text-slate-400 leading-relaxed">Pulse-based liveness verification ensuring that the subject is a biological entity, not a digital mask.</p>
          </div>
          <div className="space-y-4">
            <Lock className="text-cyan-400" size={40} />
            <h3 className="text-2xl font-bold">Dharma Security</h3>
            <p className="text-slate-400 leading-relaxed">Ethical AI frameworks mitigating demographic bias to ensure parity across all human variables.</p>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-950 border-t border-white/5 pt-20 pb-10 px-6 md:px-10">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-4 gap-12 mb-16">
            <div className="col-span-2">
              <div className="text-2xl font-black tracking-tighter mb-6 flex items-center gap-2">
                <div className="w-6 h-6 bg-cyan-500 rounded-sm" /> YANTRA
              </div>
              <p className="text-slate-500 max-w-sm mb-8">
                The world's first unified biometric reconciliation engine built for the next generation of digital sovereignty.
              </p>
              <div className="flex gap-4">
                <div className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center hover:bg-cyan-500/20 transition-colors cursor-pointer"><Globe size={18}/></div>
                <div className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center hover:bg-cyan-500/20 transition-colors cursor-pointer"><Database size={18}/></div>
              </div>
            </div>
            <div>
              <h4 className="font-bold mb-6 text-sm uppercase tracking-widest text-slate-300">Engine</h4>
              <ul className="space-y-4 text-slate-500 text-sm">
                <li className="hover:text-cyan-400 cursor-pointer">Vision Transformers</li>
                <li className="hover:text-cyan-400 cursor-pointer">rPPG Analysis</li>
                <li className="hover:text-cyan-400 cursor-pointer">Edge Deployment</li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold mb-6 text-sm uppercase tracking-widest text-slate-300">Company</h4>
              <ul className="space-y-4 text-slate-500 text-sm">
                <li className="hover:text-cyan-400 cursor-pointer">Ethics Board</li>
                <li className="hover:text-cyan-400 cursor-pointer">Security Audit</li>
                <li className="hover:text-cyan-400 cursor-pointer">Documentation</li>
              </ul>
            </div>
          </div>
          <div className="pt-8 border-t border-white/5 flex flex-col md:flex-row justify-between items-center gap-4 text-xs font-mono text-slate-600">
            <p>© 2026 YANTRA BIOMETRICS. ALL RIGHTS RESERVED.</p>
            <div className="flex gap-8">
              <span>PRIVACY_PROTOCOL_V4</span>
              <span>SYSTEM_STATUS: NOMINAL</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

function StatCard({ title, value, desc, colSpan = "" }) {
  return (
    <motion.div 
      whileHover={{ scale: 1.02, backgroundColor: "rgba(255,255,255,0.03)" }}
      className={`${colSpan} bg-white/[0.02] backdrop-blur-xl rounded-3xl p-8 border border-white/5 transition-all`}
    >
      <p className="text-[10px] font-mono text-slate-500 uppercase tracking-[0.2em] mb-4">{title}</p>
      <h2 className="text-5xl font-black text-white">{value}</h2>
      <p className="text-xs font-mono text-cyan-500/60 mt-2 uppercase">{desc}</p>
    </motion.div>
  );
}
