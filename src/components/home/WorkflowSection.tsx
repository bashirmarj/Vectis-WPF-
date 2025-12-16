import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const steps = [
  { id: "01", title: "Upload", text: "Submit your CAD files securely to our portal for instant parsing." },
  { id: "02", title: "Quote", text: "Get detailed pricing and lead time within 24 hours." },
  { id: "03", title: "Manufacture", text: "Production begins using high-speed 5-axis CNC centers." },
  { id: "04", title: "Quality", text: "Rigorous inspection ensures every part meets specifications." },
  { id: "05", title: "Deliver", text: "Parts are packed and shipped directly to your facility." },
];

const WorkflowSection: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % steps.length);
    }, 2500);

    return () => clearInterval(interval);
  }, []);

  return (
    <section className="relative z-10 py-24 bg-transparent border-t border-slate-200">
      <div className="container mx-auto px-6">
        <div className="mb-16 max-w-2xl">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">Streamlined Workflow</h2>
          <p className="text-slate-600 font-light">From digital file to physical part in record time.</p>
        </div>

        <div className="flex flex-col md:flex-row border-t border-slate-200 bg-white shadow-md rounded-sm overflow-hidden">
          {steps.map((step, index) => {
            const isActive = index === activeStep;
            
            return (
              <motion.div
                key={step.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="flex-1 p-8 border-l border-slate-200 first:border-l-0 border-b md:border-b-0 border-slate-200 cursor-default relative bg-white overflow-hidden"
              >
                {/* Gradient Sweep Overlay */}
                <motion.div 
                  className="absolute inset-0 pointer-events-none"
                  initial={{ opacity: 0 }}
                  animate={{ 
                    opacity: isActive ? 1 : 0,
                  }}
                  transition={{ duration: 0.4 }}
                >
                  {/* Sweeping gradient that moves across */}
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-primary/15 to-transparent"
                    initial={{ x: '-100%' }}
                    animate={{ 
                      x: isActive ? '100%' : '-100%',
                    }}
                    transition={{ 
                      duration: isActive ? 2.5 : 0,
                      ease: 'linear',
                    }}
                  />
                  {/* Static subtle glow background */}
                  <div className="absolute inset-0 bg-gradient-to-br from-primary/8 via-primary/5 to-transparent" />
                  {/* Glowing border effect */}
                  <div className="absolute inset-0 border-2 border-primary/20 rounded-sm" />
                </motion.div>

                <motion.span 
                  className="block text-5xl font-bold mb-6 transition-all duration-300 relative z-10"
                  animate={{
                    color: isActive ? 'hsl(var(--primary))' : 'hsl(222, 47%, 11%)',
                    x: isActive ? 8 : 0,
                  }}
                  transition={{ duration: 0.3 }}
                >
                  {step.id}
                </motion.span>
                <h3 className="text-xl font-bold text-slate-900 mb-3 uppercase tracking-wide relative z-10">{step.title}</h3>
                <p className="text-slate-600 text-sm leading-relaxed font-light relative z-10">{step.text}</p>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default WorkflowSection;
