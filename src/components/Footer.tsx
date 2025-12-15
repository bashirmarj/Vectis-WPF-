import { Link } from "react-router-dom";
import { Mail, Phone, MapPin } from "lucide-react";
import logo from "@/assets/logo.png";

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="relative z-10 bg-transparent text-white border-t border-white/10">
      <div className="container-custom py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Company Info */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <div className="bg-white rounded-lg px-2 py-1">
                <img src={logo} alt="Vectis Manufacturing Logo" className="h-10 w-auto" />
              </div>
              <span className="text-xl font-bold text-white">VECTIS</span>
            </div>
            <p className="text-gray-400 text-sm">
              Precision manufacturing solutions from design to delivery. Your trusted partner for custom engineering excellence.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="font-semibold text-lg mb-4 text-white">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <Link to="/about" className="text-gray-300 hover:text-primary transition-colors text-sm">
                  About Us
                </Link>
              </li>
              <li>
                <Link to="/services" className="text-gray-300 hover:text-primary transition-colors text-sm">
                  Services
                </Link>
              </li>
              <li>
                <Link to="/capabilities" className="text-gray-300 hover:text-primary transition-colors text-sm">
                  Capabilities
                </Link>
              </li>
              <li>
                <Link to="/projects" className="text-gray-300 hover:text-primary transition-colors text-sm">
                  Projects
                </Link>
              </li>
            </ul>
          </div>

          {/* Services */}
          <div>
            <h3 className="font-semibold text-lg mb-4 text-white">Services</h3>
            <ul className="space-y-2">
              <li className="text-gray-400 text-sm">Custom Manufacturing</li>
              <li className="text-gray-400 text-sm">Prototype Design</li>
              <li className="text-gray-400 text-sm">Custom Parts & Assemblies</li>
              <li className="text-gray-400 text-sm">Turnkey Solutions</li>
            </ul>
          </div>

          {/* Contact Info */}
          <div>
            <h3 className="font-semibold text-lg mb-4 text-white">Contact Us</h3>
            <ul className="space-y-3">
              <li className="flex items-start space-x-2 text-gray-400 text-sm">
                <Phone className="h-4 w-4 mt-0.5 flex-shrink-0" />
                <span>(123) 456-7890</span>
              </li>
              <li className="flex items-start space-x-2 text-gray-400 text-sm">
                <Mail className="h-4 w-4 mt-0.5 flex-shrink-0" />
                <span>info@vectis-mfg.com</span>
              </li>
              <li className="flex items-start space-x-2 text-gray-400 text-sm">
                <MapPin className="h-4 w-4 mt-0.5 flex-shrink-0" />
                <span>123 Industrial Way<br />Manufacturing City, ST 12345</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-white/10 mt-8 pt-8 text-center text-gray-400 text-sm">
          <p>&copy; {currentYear} Vectis Manufacturing. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
