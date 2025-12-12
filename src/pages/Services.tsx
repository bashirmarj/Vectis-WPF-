import { Link } from "react-router-dom";
import { Package, ArrowRight, CheckCircle } from "lucide-react";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import customPartsImg from "@/assets/custom-parts.jpg";
import { PartUploadForm } from "@/components/PartUploadForm";

const Services = () => {
  return (
    <div className="min-h-screen">
      <Navigation />

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 bg-accent text-accent-foreground overflow-hidden">
        <div
          className="absolute inset-0 opacity-10 bg-cover bg-center"
          style={{ backgroundImage: `url(${customPartsImg})` }}
        ></div>
        <div className="container-custom relative z-10">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-primary rounded-lg flex items-center justify-center">
              <Package className="h-8 w-8 text-primary-foreground" />
            </div>
            <h1 className="text-white mb-0">Custom Parts & Assemblies</h1>
          </div>
          <p className="text-xl text-gray-200 leading-relaxed max-w-3xl">
            Precision-manufactured components and assemblies engineered to meet your exact specifications and quality standards.
          </p>
        </div>
      </section>

      {/* Content Section */}
      <section className="section-spacing bg-background">
        <div className="container-custom">
          <div className="grid md:grid-cols-2 gap-12 items-start mb-16">
            <div>
              <img
                src={customPartsImg}
                alt="Custom Parts & Assemblies"
                className="rounded-lg shadow-2xl w-full mb-8"
              />
            </div>
            <div>
              <h2 className="mb-6">Precision Manufacturing</h2>
              <p className="text-lg text-muted-foreground mb-8 leading-relaxed">
                Our custom parts and assemblies service delivers precision-engineered components tailored to your 
                exact specifications. Using advanced CNC machining and multi-axis manufacturing capabilities, 
                we produce parts with tight tolerances and exceptional quality.
              </p>
              
              <h3 className="text-xl font-bold mb-4">Key Features</h3>
              <div className="space-y-3 mb-8">
                {[
                  "CNC machining capabilities",
                  "Multi-axis manufacturing",
                  "Tight tolerance production",
                  "Material versatility",
                  "Assembly and integration services",
                  "Upload STEP files for instant quotation",
                ].map((feature, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <CheckCircle className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-muted-foreground">{feature}</span>
                  </div>
                ))}
              </div>

              <Button size="lg" asChild>
                <Link to="/contact">
                  Request a Quote <ArrowRight className="ml-2 h-5 w-5" />
                </Link>
              </Button>
            </div>
          </div>

          {/* Upload Form Section */}
          <div className="mt-16">
            <PartUploadForm />
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Services;
