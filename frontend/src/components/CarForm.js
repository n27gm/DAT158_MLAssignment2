import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import { SelectField } from "@/components/Field";

const ACCIDENT = [
  { value: "None reported", label: "Ingen skader rapportert" },
  { value: "At least 1 accident or damage reported", label: "Minst én skade rapportert" },
];
const TITLE = [{ value: "Yes", label: "Ja" }, { value: "No", label: "Nei" }];
const START = { brand: "", model: "", model_year: "2018", milage: "50000", fuel_type: "Gasoline",
  engine: "", transmission: "A/T", ext_col: "Black", int_col: "Black",
  accident: "None reported", clean_title: "Yes" };

export const CarForm = ({ options, onSubmit, loading }) => {
  const [car, setCar] = useState(START);
  const set = (k) => (v) => setCar((c) => ({ ...c, [k]: v }));
  const models = options.models_by_brand[car.brand] || [];
  const engines = options.engines_by_model[`${car.brand}||${car.model}`] || [];
  const ready = car.brand && car.model && car.engine && car.model_year && car.milage !== "";

  const submit = (e) => {
    e.preventDefault();
    onSubmit({ ...car, model_year: Number(car.model_year), milage: Number(car.milage) });
  };

  return (
    <form onSubmit={submit} data-testid="form-used-car-estimation"
      className="bg-white border border-slate-200 rounded-2xl p-6 sm:p-8 shadow-sm grid grid-cols-1 sm:grid-cols-2 gap-5">
      <SelectField id="brand" label="Merke" value={car.brand} items={options.brands}
        onChange={(v) => setCar((c) => ({ ...c, brand: v, model: "", engine: "" }))} />
      <SelectField id="model" label="Modell" value={car.model} items={models}
        onChange={(v) => setCar((c) => ({ ...c, model: v, engine: "" }))}
        placeholder={car.brand ? "Velg modell" : "Velg merke først"} />
      <div className="space-y-1.5">
        <Label htmlFor="model_year" className="text-xs font-semibold uppercase tracking-wider text-slate-500">Årsmodell</Label>
        <Input id="model_year" type="number" min={options.year_min} max={options.year_max} value={car.model_year}
          onChange={(e) => set("model_year")(e.target.value)} data-testid="input-model-year" />
      </div>
      <div className="space-y-1.5">
        <Label htmlFor="milage" className="text-xs font-semibold uppercase tracking-wider text-slate-500">Kjørelengde (miles)</Label>
        <Input id="milage" type="number" min={0} value={car.milage}
          onChange={(e) => set("milage")(e.target.value)} data-testid="input-milage" />
      </div>
      <div className="sm:col-span-2">
        <SelectField id="engine" label="Motor" value={car.engine} items={engines} onChange={set("engine")}
          placeholder={car.model ? "Velg motor" : "Velg modell først"} />
      </div>
      <SelectField id="fuel_type" label="Drivstoff" value={car.fuel_type} items={options.fuel_types} onChange={set("fuel_type")} />
      <SelectField id="transmission" label="Girkasse" value={car.transmission} items={options.transmissions} onChange={set("transmission")} />
      <SelectField id="ext_col" label="Utvendig farge" value={car.ext_col} items={options.ext_cols} onChange={set("ext_col")} />
      <SelectField id="int_col" label="Innvendig farge" value={car.int_col} items={options.int_cols} onChange={set("int_col")} />
      <SelectField id="accident" label="Skadehistorikk" value={car.accident} items={ACCIDENT} onChange={set("accident")} />
      <SelectField id="clean_title" label="Ren tittel (clean title)" value={car.clean_title} items={TITLE} onChange={set("clean_title")} />
      <Button type="submit" disabled={!ready || loading} data-testid="submit-estimate-button"
        className="sm:col-span-2 h-12 rounded-xl bg-sky-600 hover:bg-sky-500 text-white font-bold text-base">
        {loading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />} Estimer pris
      </Button>
    </form>
  );
};
