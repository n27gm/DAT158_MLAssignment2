import { useEffect, useState } from "react";
import "@/App.css";
import axios from "axios";
import { Toaster, toast } from "sonner";
import { CarForm } from "@/components/CarForm";
import { ResultPanel } from "@/components/ResultPanel";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

export default function App() {
  const [options, setOptions] = useState(null);
  const [price, setPrice] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    axios.get(`${API}/options`).then((r) => setOptions(r.data))
      .catch(() => toast.error("Kunne ikke laste data fra serveren"));
  }, []);

  const estimate = async (car) => {
    setLoading(true);
    try {
      const r = await axios.post(`${API}/predict`, car);
      setPrice(r.data.price);
    } catch {
      toast.error("Noe gikk galt. Sjekk at alle felt er fylt ut.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900" data-testid="home-container">
      <Toaster position="top-center" />
      <header className="border-b border-slate-200 bg-white">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <span className="font-heading font-extrabold text-lg">Bruktbilpris</span>
          <span className="text-xs text-slate-500">Kaggle Playground S4E9</span>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <h1 className="font-heading text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight">
          Hva er bruktbilen verdt?
        </h1>
        <p className="mt-3 text-sm sm:text-base text-slate-600 max-w-2xl">
          Fyll inn opplysninger om bilen, så estimerer en maskinlæringsmodell (LightGBM) prisen
          basert på nesten 190 000 bilannonser.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start mt-10">
          <div className="lg:col-span-7">
            {options ? <CarForm options={options} onSubmit={estimate} loading={loading} />
              : <p className="text-sm text-slate-500" data-testid="loading-options">Laster...</p>}
          </div>
          <div className="lg:col-span-5 lg:sticky lg:top-6">
            <ResultPanel price={price} metrics={options?.metrics} />
          </div>
        </div>
      </main>
    </div>
  );
}
