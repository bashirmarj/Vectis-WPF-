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
          <p className="text-slate-900 font-light">From digital file to physical part in record time.</p>
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
                className={`flex-1 p-8 border-l border-slate-200 first:border-l-0 border-b md:border-b-0 border-slate-200 transition-all duration-500 cursor-default relative ${
                  isActive ? 'bg-slate-50' : 'bg-white hover:bg-slate-50'
                }`}
              >
                {/* Progress bar at top */}
                <motion.div 
                  className="absolute top-0 left-0 h-[2px] bg-primary"
                  initial={{ width: '0%' }}
                  animate={{ 
                    width: isActive ? '100%' : '0%',
                  }}
                  transition={{ 
                    duration: isActive ? 2.5 : 0.3,
                    ease: isActive ? 'linear' : 'easeOut'
                  }}
                />

                <motion.span 
                  className="block text-5xl font-bold mb-6 transition-all duration-300"
                  animate={{
                    color: isActive ? 'hsl(var(--primary))' : 'hsl(222, 47%, 11%)',
                    x: isActive ? 8 : 0,
                  }}
                  transition={{ duration: 0.3 }}
                >
                  {step.id}
                </motion.span>
                <h3 className="text-xl font-bold text-slate-900 mb-3 uppercase tracking-wide">{step.title}</h3>
                <p className="text-slate-600 text-sm leading-relaxed font-light">{step.text}</p>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default WorkflowSection;
