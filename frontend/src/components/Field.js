import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export const SelectField = ({ id, label, value, onChange, items, placeholder = "Velg..." }) => (
  <div className="space-y-1.5">
    <Label htmlFor={id} className="text-xs font-semibold uppercase tracking-wider text-slate-500">{label}</Label>
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger id={id} data-testid={`select-${id}`} className="bg-white">
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent className="max-h-72">
        {items.map((it) => {
          const v = typeof it === "string" ? it : it.value;
          const l = typeof it === "string" ? it : it.label;
          return <SelectItem key={v} value={v}>{l}</SelectItem>;
        })}
      </SelectContent>
    </Select>
  </div>
);
