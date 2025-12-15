import { Link } from "react-router-dom";
import { ArrowRight, CheckCircle } from "lucide-react";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import AnimatedSection from "@/components/home/AnimatedSection";
import ParticleBackground from "@/components/home/ParticleBackground";
import precisionManufacturingImg from "@/assets/precision-manufacturing.png";
import { PartUploadForm } from "@/components/PartUploadForm";
const Services = () => {
  return <div className="min-h-screen bg-black">
      <Navigation />
      <ParticleBackground />

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 overflow-hidden z-10">
        <div className="container-custom relative z-10">
          <AnimatedSection animation="fadeUp">
            <h1 className="text-5xl md:text-6xl font-bold text-white mb-6 leading-snug tracking-normal">Custom Parts For Production & Prototyping</h1>
            <p className="text-xl text-gray-300 leading-relaxed max-w-3xl">Precision-manufactured components engineered to meet your exact specifications.</p>
          </AnimatedSection>
        </div>
      </section>

      {/* Content Section */}
      <section className="relative z-10 border-t border-white/5">
        <div className="container-custom section-spacing">
          <div className="grid md:grid-cols-2 gap-12 items-start mb-16">
            <AnimatedSection animation="fadeRight">
              <img src={precisionManufacturingImg} alt="Precision Manufacturing" className="rounded-lg shadow-2xl w-full mb-8 border border-white/10" />
            </AnimatedSection>
            <AnimatedSection animation="fadeLeft" delay={100}>
              <div className="flex items-start gap-2 mb-4">
                <div className="w-1 h-8 bg-primary rounded-full"></div>
                <h2 className="text-3xl font-bold text-white tracking-tight font-mono uppercase">
                  Precision Manufacturing
                </h2>
              </div>
              <p className="text-lg text-gray-300 mb-8 leading-relaxed">
                Our custom parts and assemblies service delivers precision-engineered components tailored to your 
                exact specifications. Using advanced CNC machining and multi-axis manufacturing capabilities, 
                we produce parts with tight tolerances and exceptional quality.
              </p>
              
              <div className="flex items-start gap-2 mb-4">
                <div className="w-1 h-8 bg-primary rounded-full"></div>
                <h3 className="text-xl font-bold text-white tracking-tight font-mono uppercase">
                  Key Features
                </h3>
              </div>
              <div className="space-y-3 mb-8">
                {["CNC machining capabilities", "Multi-axis manufacturing", "Tight tolerance production", "Material versatility", "Assembly and integration services", "Upload STEP files for instant quotation"].map((feature, idx) => <AnimatedSection key={idx} animation="fadeLeft" delay={150 + idx * 50}>
                    <div className="flex items-start gap-3">
                      <CheckCircle className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                      <span className="text-gray-300">{feature}</span>
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