import { Link } from "react-router-dom";
import { ArrowRight, CheckCircle } from "lucide-react";
import { motion } from "framer-motion";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import AnimatedSection from "@/components/home/AnimatedSection";
import ParticleBackground from "@/components/home/ParticleBackground";
import precisionManufacturingImg from "@/assets/precision-manufacturing.png";
import { PartUploadForm } from "@/components/PartUploadForm";
const Services = () => {
  return <div className="min-h-screen bg-white">
      <Navigation />
      <ParticleBackground />

      {/* Hero Section */}
      <section className="relative pt-32 pb-8 overflow-hidden z-10">
        <div className="container-custom relative z-10">
          {/* Animated title */}
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-5xl md:text-6xl font-bold text-slate-900 mb-6 leading-snug tracking-normal uppercase"
          >
            Custom Parts For{' '}
            <motion.span 
              className="inline-block text-transparent bg-clip-text"
              style={{
                backgroundImage: 'linear-gradient(90deg, #0f172a 0%, #0f172a 5%, hsl(var(--primary)) 15%, hsl(var(--primary)) 35%, #0f172a 45%, #0f172a 50%, #0f172a 50%, #0f172a 55%, hsl(var(--primary)) 65%, hsl(var(--primary)) 85%, #0f172a 95%, #0f172a 100%)',
                backgroundSize: '200% auto',
              }}
              animate={{ backgroundPositionX: ['0%', '-100%'] }}
              transition={{ duration: 5, repeat: Infinity, ease: "linear" }}
            >
              Production & Prototyping
            </motion.span>
          </motion.h1>

          {/* Animated subtitle */}
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.4 }}
            className="text-xl text-slate-600 leading-relaxed max-w-3xl"
          >
            Precision-manufactured components engineered to meet your exact specifications.
          </motion.p>
        </div>
      </section>

      {/* Content Section */}
      <section className="relative z-10 border-t border-slate-200">
        <div className="container-custom py-8 md:py-12">
          <div className="grid md:grid-cols-2 gap-12 items-start mb-16">
            <AnimatedSection animation="fadeRight">
              <img src={precisionManufacturingImg} alt="Precision Manufacturing" className="rounded-lg shadow-md w-full mb-8 border border-slate-200" />
            </AnimatedSection>
            <AnimatedSection animation="fadeLeft" delay={100}>
              <div className="flex items-start gap-2 mb-4">
                <div className="w-1 h-8 bg-primary rounded-full"></div>
                <h2 className="text-3xl font-bold text-slate-900 tracking-tight font-mono uppercase">
                  Precision Manufacturing
                </h2>
              </div>
              <p className="text-lg text-slate-600 mb-8 leading-relaxed">
                Our custom parts and assemblies service delivers precision-engineered components tailored to your 
                exact specifications. Using advanced CNC machining and multi-axis manufacturing capabilities, 
                we produce parts with tight tolerances and exceptional quality.
              </p>
              
              <div className="flex items-start gap-2 mb-4">
                <div className="w-1 h-8 bg-primary rounded-full"></div>
                <h3 className="text-xl font-bold text-slate-900 tracking-tight font-mono uppercase">
                  Key Features
                </h3>
              </div>
              <div className="space-y-3 mb-8">
                {["CNC machining capabilities", "Multi-axis manufacturing", "Tight tolerance production", "Material versatility", "Assembly and integration services", "Upload STEP files for instant quotation"].map((feature, idx) => <AnimatedSection key={idx} animation="fadeLeft" delay={150 + idx * 50}>
                    <div className="flex items-start gap-3">
                      <CheckCircle className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                      <span className="text-slate-600">{feature}</span>
                    </div>
                  </AnimatedSection>)}
              </div>

              <Button size="lg" asChild>
                <Link to="/contact">
                  Request a Quote <ArrowRight className="ml-2 h-5 w-5" />
                </Link>
              </Button>
            </AnimatedSection>
          </div>

          {/* Upload Form Section */}
          <AnimatedSection animation="fadeUp" delay={200}>
            <div className="mt-16">
              <PartUploadForm />
            </div>
          </AnimatedSection>
        </div>
      </section>

      <Footer />
    </div>;
};
export default Services;
