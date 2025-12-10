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
import CapabilityCard from "@/components/home/CapabilityCard";
import { useScrollAnimation, useStaggerAnimation } from "@/hooks/useScrollAnimation";
import heroImage from "@/assets/hero-cnc-turbine.png";
import customPartsImg from "@/assets/custom-parts.jpg";
import prototypeImg from "@/assets/prototype-design.jpg";
import turnkeyImg from "@/assets/turnkey-solutions.jpg";
import capabilitiesImg from "@/assets/capabilities-bg.jpg";

const Index = () => {
  const services = [
    {
      icon: Wrench,
      title: "Prototype Design",
      description: "Transform your concepts into functional prototypes with our expert engineering team and rapid turnaround.",
      image: prototypeImg,
      link: "/services/prototype-design",
    },
    {
      icon: Package,
      title: "Custom Parts",
      description: "High-quality custom components manufactured to meet your exact specifications and tolerances.",
      image: customPartsImg,
      link: "/services/custom-parts",
    },
    {
      icon: TrendingUp,
      title: "Turnkey Solutions",
      description: "Complete project lifecycle management from initial design through final delivery and beyond.",
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
    { icon: Cpu, title: "CNC Machining", desc: "5-axis precision milling and turning" },
    { icon: Zap, title: "Wire EDM", desc: "Complex geometries with tight tolerances" },
    { icon: Layers, title: "Sheet Metal", desc: "Laser cutting, bending, and forming" },
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

  const { ref: servicesRef, isVisible: servicesVisible } = useStaggerAnimation(services.length, 150);
  const { ref: whyRef, isVisible: whyVisible } = useStaggerAnimation(whyChooseUs.length, 200);

  return (
    <div className="min-h-screen overflow-x-hidden">
      <Navigation />

      {/* Hero Section - Full Viewport with Parallax, extends behind navigation */}
      <section className="relative h-screen flex flex-col justify-center overflow-hidden -mt-16 pt-16">
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
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20">
          <ScrollIndicator />
        </div>

        {/* Gradient fade into next section */}
        <div className="absolute bottom-0 left-0 right-0 h-40 bg-gradient-to-b from-transparent to-background z-10" />
      </section>

      {/* Services Overview */}
      <section className="py-20 md:py-32">
        <div className="container-custom">
          <AnimatedSection className="text-center max-w-3xl mx-auto mb-16">
            <p className="text-primary font-semibold text-sm uppercase tracking-[0.2em] mb-4">
              What We Offer
            </p>
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-6">
              Our Core Services
            </h2>
            <p className="text-lg text-muted-foreground">
              Comprehensive manufacturing solutions backed by decades of engineering expertise 
              and state-of-the-art technology.
            </p>
          </AnimatedSection>

          <div ref={servicesRef} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {services.map((service, index) => (
              <CapabilityCard
                key={index}
                title={service.title}
                description={service.description}
                image={service.image}
                icon={service.icon}
                link={service.link}
                isVisible={servicesVisible}
                delay={index * 150}
              />
            ))}
          </div>
        </div>
      </section>

      {/* Capabilities Strip */}
      <section className="py-16 border-t border-border/30">
        <div className="container-custom">
          <AnimatedSection className="text-center mb-12">
            <p className="text-primary font-semibold text-sm uppercase tracking-[0.2em] mb-3">
              Manufacturing Capabilities
            </p>
            <h3 className="text-2xl md:text-3xl font-bold">
              Precision at Every Step
            </h3>
          </AnimatedSection>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {capabilities.map((cap, index) => {
              const Icon = cap.icon;
              return (
                <AnimatedSection
                  key={index}
                  animation="scaleIn"
                  delay={index * 100}
                  className="flex items-center gap-4 p-6 bg-background rounded-lg border border-border hover:border-primary/50 transition-colors group"
                >
                  <div className="w-14 h-14 bg-primary/10 rounded-lg flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                    <Icon className="w-7 h-7 text-primary" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-lg">{cap.title}</h4>
                    <p className="text-sm text-muted-foreground">{cap.desc}</p>
                  </div>
                </AnimatedSection>
              );
            })}
          </div>

          <AnimatedSection className="text-center mt-10">
            <Button variant="outline" asChild>
              <Link to="/capabilities">
                View All Capabilities <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </AnimatedSection>
        </div>
      </section>

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
