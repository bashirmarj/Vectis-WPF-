import { Link } from "react-router-dom";
import { ArrowRight, Wrench, Package, TrendingUp, Cpu, Zap, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import AnimatedSection from "@/components/home/AnimatedSection";
import AnimatedCounter from "@/components/home/AnimatedCounter";
import ParallaxSection from "@/components/home/ParallaxSection";
import ProcessTimeline from "@/components/home/ProcessTimeline";
import ScrollIndicator from "@/components/home/ScrollIndicator";
import { useStaggerAnimation } from "@/hooks/useScrollAnimation";
import heroImage from "@/assets/hero-cnc-turbine.png";
import customPartsImg from "@/assets/custom-parts.jpg";
import prototypeImg from "@/assets/prototype-cnc-part.png";
import turnkeyImg from "@/assets/turnkey-solutions.jpg";
import capabilitiesImg from "@/assets/capabilities-bg.jpg";

const Index = () => {
  const services = [
    {
      icon: Wrench,
      title: "Prototype Design",
      description: "Transform your concepts into functional prototypes with our expert engineering team and rapid turnaround. We bridge the gap between design and production.",
      image: prototypeImg,
      link: "/services/prototype-design",
    },
    {
      icon: Package,
      title: "Custom Parts",
      description: "High-quality custom components manufactured to meet your exact specifications and tolerances. From one-off pieces to production runs.",
      image: customPartsImg,
      link: "/services/custom-parts",
    },
    {
      icon: TrendingUp,
      title: "Turnkey Solutions",
      description: "Complete project lifecycle management from initial design through final delivery and beyond. We handle everything so you can focus on your business.",
      image: turnkeyImg,
      link: "/services/turnkey-solutions",
    },
  ];

  const stats = [
    { value: "25+", label: "Years Experience" },
    { value: "500+", label: "Projects Completed" },
    { value: "98%", label: "On-Time Delivery" },
    { value: "ISO", label: "9001 Certified" },
  ];

  const capabilities = [
    { 
      icon: Cpu, 
      title: "CNC Machining", 
      desc: "5-axis precision milling and turning for complex geometries with tolerances down to ±0.001\"",
      image: capabilitiesImg // Placeholder - replace with CNC-specific image
    },
    { 
      icon: Zap, 
      title: "Wire EDM", 
      desc: "Complex geometries with tight tolerances. Ideal for hardened materials and intricate details.",
      image: capabilitiesImg // Placeholder - replace with Wire EDM image
    },
    { 
      icon: Layers, 
      title: "Sheet Metal", 
      desc: "Laser cutting, bending, and forming. From prototypes to high-volume production.",
      image: capabilitiesImg // Placeholder - replace with Sheet Metal image
    },
  ];

  const whyChooseUs = [
    {
      number: "01",
      title: "End-to-End Solutions",
      description: "From initial concept and design through manufacturing and final delivery, we manage your entire project lifecycle.",
    },
    {
      number: "02",
      title: "Precision Engineering",
      description: "State-of-the-art equipment and rigorous quality control ensure every part meets exact specifications.",
    },
    {
      number: "03",
      title: "Expert Team",
      description: "Our experienced engineers and technicians bring deep industry knowledge to every project.",
    },
  ];

  const { ref: whyRef, isVisible: whyVisible } = useStaggerAnimation(whyChooseUs.length, 200);

  return (
    <div className="min-h-screen overflow-x-hidden">
      <Navigation />

      {/* Hero Section - Full Viewport with Parallax */}
      <section className="relative min-h-[120vh] flex flex-col justify-center overflow-hidden -mt-16 pt-16">
        <ParallaxSection
          backgroundImage={heroImage}
          speed={0.2}
          overlayOpacity={0.75}
          className="absolute inset-0"
          contentClassName="hidden"
        >
          <div />
        </ParallaxSection>
        
        <div className="container-custom relative z-10 flex-1 flex items-center">
          <div className="max-w-4xl">
            <AnimatedSection animation="fadeUp" delay={0}>
              <p className="text-primary font-semibold text-sm md:text-base mb-4 uppercase tracking-[0.2em]">
                Precision Manufacturing Excellence
              </p>
            </AnimatedSection>
            
            <AnimatedSection animation="fadeUp" delay={150}>
              <h1 className="text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-bold text-white mb-6 leading-[1.1] tracking-tight">
                Custom Manufacturing
                <br />
                <span className="text-primary">From Concept</span>
                <br />
                to Completion
              </h1>
            </AnimatedSection>
            
            <AnimatedSection animation="fadeUp" delay={300}>
              <p className="text-lg md:text-xl text-gray-300 mb-10 max-w-2xl leading-relaxed">
                Vectis delivers turnkey manufacturing projects, precision prototype designs, 
                and custom parts engineered to perfection.
              </p>
            </AnimatedSection>
            
            <AnimatedSection animation="fadeUp" delay={450}>
              <div className="flex flex-col sm:flex-row gap-4">
                <Button size="lg" className="group" asChild>
                  <Link to="/contact">
                    Request a Quote 
                    <ArrowRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1" />
                  </Link>
                </Button>
                <Button size="lg" variant="outline-light" asChild>
                  <Link to="/capabilities">Explore Capabilities</Link>
                </Button>
              </div>
            </AnimatedSection>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-24 left-1/2 -translate-x-1/2 z-20">
          <ScrollIndicator />
        </div>

        {/* Gradient fade into next section */}
        <div className="absolute bottom-0 left-0 right-0 h-48 bg-gradient-to-b from-transparent to-background z-10" />
      </section>

      {/* Services - Full-Bleed Vertical Sections */}
      {services.map((service, index) => {
        const Icon = service.icon;
        const isEven = index % 2 === 0;
        
        return (
          <section key={index} className="relative min-h-screen flex items-center">
            {/* Background Image with Parallax */}
            <ParallaxSection
              backgroundImage={service.image}
              speed={0.15}
              overlayOpacity={0.7}
              className="absolute inset-0"
              contentClassName="hidden"
            >
              <div />
            </ParallaxSection>

            {/* Content */}
            <div className="container-custom relative z-10">
              <div className={`max-w-xl ${isEven ? '' : 'ml-auto'}`}>
                <AnimatedSection animation={isEven ? "fadeRight" : "fadeLeft"}>
                  <div className="w-16 h-16 bg-primary/20 backdrop-blur-sm rounded-xl flex items-center justify-center mb-8 border border-primary/30">
                    <Icon className="w-8 h-8 text-primary" />
                  </div>
                </AnimatedSection>
                
                <AnimatedSection animation="fadeUp" delay={100}>
                  <p className="text-primary font-semibold text-sm uppercase tracking-[0.2em] mb-4">
                    {`0${index + 1}`} — Services
                  </p>
                </AnimatedSection>
                
                <AnimatedSection animation="fadeUp" delay={200}>
                  <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-6">
                    {service.title}
                  </h2>
                </AnimatedSection>
                
                <AnimatedSection animation="fadeUp" delay={300}>
                  <p className="text-lg md:text-xl text-gray-300 mb-10 leading-relaxed">
                    {service.description}
                  </p>
                </AnimatedSection>
                
                <AnimatedSection animation="fadeUp" delay={400}>
                  <Button size="lg" variant="outline-light" className="group" asChild>
                    <Link to={service.link}>
                      Learn More 
                      <ArrowRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1" />
                    </Link>
                  </Button>
                </AnimatedSection>
              </div>
            </div>

            {/* Gradient fade at bottom */}
            <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-b from-transparent to-background z-10" />
          </section>
        );
      })}

      {/* Capabilities - Full-Bleed Vertical Sections */}
      {capabilities.map((capability, index) => {
        const Icon = capability.icon;
        const isEven = index % 2 === 0;
        
        return (
          <section key={index} className="relative min-h-[80vh] flex items-center">
            {/* Background Image with Parallax */}
            <ParallaxSection
              backgroundImage={capability.image}
              speed={0.1}
              overlayOpacity={0.8}
              className="absolute inset-0"
              contentClassName="hidden"
            >
              <div />
            </ParallaxSection>

            {/* Content */}
            <div className="container-custom relative z-10">
              <div className={`max-w-xl ${isEven ? 'ml-auto' : ''}`}>
                <AnimatedSection animation={isEven ? "fadeLeft" : "fadeRight"}>
                  <div className="w-14 h-14 bg-primary/20 backdrop-blur-sm rounded-lg flex items-center justify-center mb-6 border border-primary/30">
                    <Icon className="w-7 h-7 text-primary" />
                  </div>
                </AnimatedSection>
                
                <AnimatedSection animation="fadeUp" delay={100}>
                  <p className="text-primary font-semibold text-sm uppercase tracking-[0.2em] mb-3">
                    Capabilities
                  </p>
                </AnimatedSection>
                
                <AnimatedSection animation="fadeUp" delay={200}>
                  <h3 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-4">
                    {capability.title}
                  </h3>
                </AnimatedSection>
                
                <AnimatedSection animation="fadeUp" delay={300}>
                  <p className="text-lg text-gray-300 mb-8 leading-relaxed">
                    {capability.desc}
                  </p>
                </AnimatedSection>
                
                <AnimatedSection animation="fadeUp" delay={400}>
                  <Button variant="outline-light" asChild>
                    <Link to="/capabilities">
                      View Details <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </AnimatedSection>
              </div>
            </div>

            {/* Gradient fade at bottom */}
            <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-b from-transparent to-background z-10" />
          </section>
        );
      })}

      {/* Process Timeline */}
      <section className="py-20 md:py-32">
        <div className="container-custom">
          <AnimatedSection className="text-center max-w-3xl mx-auto mb-16">
            <p className="text-primary font-semibold text-sm uppercase tracking-[0.2em] mb-4">
              How It Works
            </p>
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-6">
              Our Process
            </h2>
            <p className="text-lg text-muted-foreground">
              From initial upload to final delivery, we streamline every step 
              of your manufacturing journey.
            </p>
          </AnimatedSection>

          <ProcessTimeline className="max-w-5xl mx-auto" />
        </div>
      </section>

      {/* Why Choose Vectis */}
      <section className="py-20 md:py-32">
        <div className="container-custom">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-20 items-center">
            <div>
              <AnimatedSection animation="fadeLeft">
                <p className="text-primary font-semibold text-sm uppercase tracking-[0.2em] mb-4">
                  Why Vectis
                </p>
                <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-8">
                  Built on Precision,
                  <br />
                  Driven by Excellence
                </h2>
              </AnimatedSection>

              <div ref={whyRef} className="space-y-8">
                {whyChooseUs.map((item, index) => (
                  <div
                    key={index}
                    className={`flex gap-5 transition-all duration-700 ${
                      whyVisible
                        ? 'opacity-100 translate-x-0'
                        : 'opacity-0 -translate-x-8'
                    }`}
                    style={{ transitionDelay: `${index * 200}ms` }}
                  >
                    <div className="flex-shrink-0 w-14 h-14 bg-primary rounded-lg flex items-center justify-center">
                      <span className="text-primary-foreground font-bold text-lg">
                        {item.number}
                      </span>
                    </div>
                    <div>
                      <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                      <p className="text-muted-foreground leading-relaxed">
                        {item.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <AnimatedSection animation="fadeUp" delay={600} className="mt-10">
                <Button size="lg" asChild>
                  <Link to="/about">Learn About Us</Link>
                </Button>
              </AnimatedSection>
            </div>

            <AnimatedSection animation="fadeRight" className="relative">
              <div className="relative rounded-lg overflow-hidden shadow-2xl">
                <img
                  src={turnkeyImg}
                  alt="Manufacturing facility"
                  className="w-full h-auto"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-accent/40 to-transparent" />
              </div>
              {/* Floating accent element */}
              <div className="absolute -bottom-6 -left-6 w-32 h-32 bg-primary/10 rounded-lg -z-10" />
              <div className="absolute -top-6 -right-6 w-24 h-24 bg-primary/5 rounded-lg -z-10" />
            </AnimatedSection>
          </div>
        </div>
      </section>

      {/* CTA Section with Parallax */}
      <ParallaxSection
        backgroundImage={capabilitiesImg}
        speed={0.15}
        overlayOpacity={0.85}
        className="py-24 md:py-40"
      >
        <div className="container-custom text-center">
          <AnimatedSection animation="fadeUp">
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-6">
              Ready to Start Your Project?
            </h2>
          </AnimatedSection>
          
          <AnimatedSection animation="fadeUp" delay={150}>
            <p className="text-xl text-gray-300 mb-10 max-w-2xl mx-auto">
              Let's discuss how Vectis can bring your manufacturing vision to life 
              with precision and expertise.
            </p>
          </AnimatedSection>
          
          <AnimatedSection animation="fadeUp" delay={300}>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" className="animate-pulse-glow" asChild>
                <Link to="/contact">
                  Get a Quote <ArrowRight className="ml-2 h-5 w-5" />
                </Link>
              </Button>
              <Button size="lg" variant="outline-light" asChild>
                <Link to="/projects">View Our Work</Link>
              </Button>
            </div>
          </AnimatedSection>
        </div>
      </ParallaxSection>

      {/* Stats Section - At bottom before footer */}
      <section className="py-16 bg-accent/50 border-t border-border/30">
        <div className="container-custom">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8">
            {stats.map((stat, index) => (
              <AnimatedCounter
                key={index}
                value={stat.value}
                label={stat.label}
                delay={index * 100}
              />
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Index;
