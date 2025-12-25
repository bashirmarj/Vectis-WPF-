import { Target, Award, Users, TrendingUp } from "lucide-react";
import { useLayoutEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Card, CardContent } from "@/components/ui/card";
import AnimatedSection from "@/components/home/AnimatedSection";
import ParticleBackground from "@/components/home/ParticleBackground";
import missionHeroImg from "@/assets/mission-hero.png";

const About = () => {
  const textRef = useRef<HTMLDivElement>(null);
  const [imageHeight, setImageHeight] = useState<number | undefined>(undefined);

  useLayoutEffect(() => {
    const textEl = textRef.current;
    if (!textEl) return;

    const updateHeight = () => {
      const isDesktop = window.matchMedia('(min-width: 768px)').matches;
      if (isDesktop) {
        setImageHeight(textEl.offsetHeight);
      } else {
        setImageHeight(undefined);
      }
    };

    const resizeObserver = new ResizeObserver(updateHeight);
    resizeObserver.observe(textEl);
    window.addEventListener('resize', updateHeight);
    updateHeight();

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('resize', updateHeight);
    };
  }, []);

  const values = [{
    icon: Target,
    title: "Precision First",
    description: "Every component manufactured to exacting standards with rigorous quality control."
  }, {
    icon: Award,
    title: "Engineering Excellence",
    description: "Decades of expertise combined with cutting-edge manufacturing technology."
  }, {
    icon: Users,
    title: "Customer Partnership",
    description: "Collaborative approach ensuring your vision becomes reality on time and on budget."
  }, {
    icon: TrendingUp,
    title: "Continuous Innovation",
    description: "Investing in the latest technology and techniques to deliver superior results."
  }];

  return <div className="min-h-screen bg-[#F5F5F2]">
      <Navigation />
      <ParticleBackground />

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 overflow-hidden z-10">
        <div className="container-custom relative z-10">
          <AnimatedSection animation="fadeUp">
            <div className="max-w-3xl">
              <h1 className="text-3xl md:text-5xl lg:text-6xl font-bold text-black leading-tight mb-8 uppercase tracking-tight">
                About <br />
                <motion.span 
                  className="relative inline-block text-transparent bg-clip-text"
                  style={{
                    backgroundImage: 'linear-gradient(90deg, #0f172a 0%, #475569 5%, #dc2626 15%, #dc2626 35%, #475569 45%, #0f172a 50%, #0f172a 50%, #475569 55%, #dc2626 65%, #dc2626 85%, #475569 95%, #0f172a 100%)',
                    backgroundSize: '200% auto',
                  }}
                  animate={{ backgroundPositionX: ['0%', '-100%'] }}
                  transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                >
                  Vectis
                </motion.span> <br />
                <span className="text-black">
                  Manufacturing
                </span>
              </h1>
              <p className="text-xl text-black/80 leading-relaxed">
                For over 25 years, Vectis has been at the forefront of precision manufacturing, delivering innovative solutions that drive our clients' success.
              </p>
              
              {/* Brand Meaning Callout */}
              <div className="mt-8 border-l-4 border-primary bg-white/60 backdrop-blur-md rounded-r-lg p-6 max-w-2xl">
                <h3 className="text-sm font-semibold text-primary mb-3 tracking-wide uppercase font-mono">The Vectis Name</h3>
                <p className="text-black/80 italic leading-relaxed">
                  Derived from the Latin word for lever, Vectis embodies precision, strength, and intelligent design. Like a lever that multiplies force, Vectis Manufacturing transforms engineering expertise into powerful, efficient solutions that drive performance and reliability.
                </p>
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>

      {/* Mission & Vision */}
      <section className="relative z-10 border-t border-gray-200">
        <div className="container-custom section-spacing">
          <div className="grid md:grid-cols-2 gap-12 items-start">
            <AnimatedSection animation="fadeRight">
              <div 
                className="relative rounded-lg shadow-xl border border-gray-200 overflow-hidden"
                style={{ height: imageHeight ? `${imageHeight}px` : 'auto' }}
              >
                <img 
                  src={missionHeroImg} 
                  alt="Precision CNC machined part with engineering drawings" 
                  className="w-full h-full object-cover object-center" 
                />
                <div className="absolute inset-0 bg-[#F5F5F2]/30"></div>
              </div>
            </AnimatedSection>
            <AnimatedSection animation="fadeLeft" delay={100}>
              <div ref={textRef} className="space-y-8">
                <div>
                  <div className="flex items-start gap-2 mb-4">
                    <div className="w-1 h-8 bg-primary rounded-full"></div>
                    <h2 className="text-3xl font-bold text-black tracking-tight font-mono uppercase">Our Mission</h2>
                  </div>
                  <p className="text-lg text-black/80 leading-relaxed">
                    To deliver exceptional manufacturing solutions that exceed our clients' expectations through precision engineering, innovative technology, and unwavering commitment to quality. We transform concepts into reality, supporting businesses across industries with reliable, scalable manufacturing partnerships.
                  </p>
                </div>
                <div>
                  <div className="flex items-start gap-2 mb-4">
                    <div className="w-1 h-8 bg-primary rounded-full"></div>
                    <h2 className="text-3xl font-bold text-black tracking-tight font-mono uppercase">Our Vision</h2>
                  </div>
                  <p className="text-lg text-black/80 leading-relaxed">
                    To be the preferred manufacturing partner for companies seeking precision, reliability, and innovation. We envision a future where advanced manufacturing technology and expert craftsmanship combine to solve complex engineering challenges and drive industrial progress.
                  </p>
                </div>
              </div>
            </AnimatedSection>
          </div>
        </div>
      </section>

      {/* Core Values */}
      <section className="relative z-10 border-t border-gray-200">
        <div className="container-custom section-spacing">
          <AnimatedSection animation="fadeUp">
            <div className="text-center max-w-3xl mx-auto mb-16">
              <div className="flex items-start gap-2 mb-4 justify-center">
                <div className="w-1 h-8 bg-primary rounded-full"></div>
                <h2 className="text-3xl font-bold text-black tracking-tight font-mono uppercase">Our Core Values</h2>
              </div>
              <p className="text-lg text-black/70">
                The principles that guide every project and partnership we undertake.
              </p>
            </div>
          </AnimatedSection>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {values.map((value, index) => {
            const Icon = value.icon;
            return <AnimatedSection key={index} animation="fadeUp" delay={index * 100}>
                  <Card className="h-full bg-[#F5F5F2] backdrop-blur-md border border-gray-200 hover:border-primary/50 transition-all duration-300 hover:-translate-y-1">
                    <CardContent className="p-6">
                      <div className="w-14 h-14 bg-primary/20 rounded-lg flex items-center justify-center mb-4">
                        <Icon className="h-7 w-7 text-primary" />
                      </div>
                      <h3 className="text-xl font-bold text-black mb-2">{value.title}</h3>
                      <p className="text-black/80">{value.description}</p>
                    </CardContent>
                  </Card>
                </AnimatedSection>;
          })}
          </div>
        </div>
      </section>

      {/* Company Stats */}
      <section className="relative z-10 border-t border-gray-200">
        
      </section>

      {/* Expertise Section */}
      <section className="relative z-10 border-t border-gray-200">
        <div className="container-custom section-spacing">
          <AnimatedSection animation="fadeUp">
            <div className="max-w-4xl">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-1 bg-primary rounded-full"></div>
                <span className="text-sm font-mono tracking-[0.2em] text-black uppercase">Our Expertise</span>
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-black mb-4">Across Industries</h2>
              <p className="text-lg text-black/70 mb-12">
                Our diverse experience spans multiple sectors, giving us unique insights into the specific challenges and requirements of each industry we serve.
              </p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {["Aerospace", "Automotive", "Medical Devices", "Industrial Equipment", "Energy & Power", "Defense & Military"].map((industry, index) => <AnimatedSection key={industry} animation="scaleIn" delay={index * 50}>
                    <div className="bg-[#F5F5F2] backdrop-blur-md border border-gray-200 rounded-lg p-4 font-semibold text-center text-black/80 hover:border-primary/50 hover:text-primary transition-colors">
                      {industry}
                    </div>
                  </AnimatedSection>)}
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>

      <Footer />
    </div>;
};
export default About;