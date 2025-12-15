import React from 'react';
import { ArrowRight, ChevronRight } from 'lucide-react';
import { motion } from 'framer-motion';

const Hero: React.FC = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-start pt-20 pb-20 z-10">
      <div className="container mx-auto px-6">
        <div className="max-w-4xl">
          <motion.div 
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: "auto", opacity: 1 }}
            transition={{ duration: 0.8, ease: "circOut" }}
            className="flex items-center gap-3 mb-6 overflow-hidden whitespace-nowrap"
          >
            <div className="w-8 h-1 bg-vectis-red shrink-0"></div>
            <motion.span 
              initial={{ color: "#dc2626" }}
              animate={{ color: ["#dc2626", "#ffffff", "#dc2626"] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              className="font-semibold tracking-[0.2em] uppercase text-sm"
            >
              Advanced Manufacturing Solutions
            </motion.span>
          </motion.div>

          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-3xl md:text-5xl lg:text-6xl font-bold text-white leading-tight mb-8 uppercase tracking-tight"
          >
            Custom <br />
            {/* Seamless Gradient Animation */}
            {/* Logic: Gradient contains 2 identical cycles. Background size is 200%. 
                0% pos shows Cycle 1. 100% pos shows Cycle 2. Since Cycle 1 == Cycle 2, it loops perfectly. */}
            <motion.span 
              className="relative inline-block text-transparent bg-clip-text"
              style={{
                backgroundImage: 'linear-gradient(90deg, #ffffff 0%, #a3a3a3 5%, #dc2626 15%, #dc2626 35%, #a3a3a3 45%, #ffffff 50%, #ffffff 50%, #a3a3a3 55%, #dc2626 65%, #dc2626 85%, #a3a3a3 95%, #ffffff 100%)',
                backgroundSize: '200% auto',
              }}
              animate={{ backgroundPositionX: ['0%', '-100%'] }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
            >
              Manufacturing
            </motion.span> <br />
            {/* Increased tracking to fix letter interference */}
            <span className="text-transparent stroke-white tracking-[0.25em]" style={{ WebkitTextStroke: '1px rgba(255,255,255,0.4)' }}>
              From Concept
            </span> <br />
            To Completion
          </motion.h1>

          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.6 }}
            className="text-vectis-muted text-lg md:text-xl max-w-2xl mb-10 leading-relaxed font-light"
          >
            Precision engineering and turnkey manufacturing solutions. We combine traditional craftsmanship with AI-driven automation to deliver parts faster.
          </motion.p>

          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="flex flex-col sm:flex-row gap-4"
          >
            <motion.a 
              href="#contact" 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="group bg-vectis-red hover:bg-vectis-redHover text-white px-8 py-4 rounded-sm font-bold uppercase tracking-widest text-sm transition-colors flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(220,38,38,0.4)]"
            >
              Request A Quote
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </motion.a>
            <motion.a 
              href="#services" 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="group bg-transparent border border-white/20 hover:bg-white/5 text-white px-8 py-4 rounded-sm font-bold uppercase tracking-widest text-sm transition-colors flex items-center justify-center gap-2"
            >
              Explore Capabilities
              <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </motion.a>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
