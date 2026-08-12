import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';

// Pages
import Home from './pages/Home';
import Analyze from './pages/Analyze';
import Hospitals from './pages/Hospitals';
import Resources from './pages/Resources';
import About from './pages/About';
import Contact from './pages/Contact';
import SkinBot from './pages/SkinBot';

/** Returns the stored user object, or null if not logged in */
const getUser = () => {
  try {
    const raw = localStorage.getItem('user');
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
};

/**
 * Guards a route behind authentication.
 * If the user is not logged in, redirects to / with the attempted path
 * stored in location.state so the auth modal can redirect after login.
 */
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const user = getUser();

  if (!user) {
    return <Navigate to="/" state={{ from: location, openAuth: true }} replace />;
  }

  return <>{children}</>;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public */}
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/contact" element={<Contact />} />

        {/* Protected — must be logged in */}
        <Route path="/analyze"   element={<ProtectedRoute><Analyze /></ProtectedRoute>} />
        <Route path="/hospitals" element={<ProtectedRoute><Hospitals /></ProtectedRoute>} />
        <Route path="/resources" element={<ProtectedRoute><Resources /></ProtectedRoute>} />
        <Route path="/skinbot"   element={<ProtectedRoute><SkinBot /></ProtectedRoute>} />
      </Routes>
    </BrowserRouter>
  );
}
