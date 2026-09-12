import { Button } from "@/components/ui/button";

/** Jump to actual recorded phase transitions; never infer new actions. */
export default function ReplayChapters({samples, seek}: {samples: {t:number;phase?:string}[]; seek:(t:number)=>void}) {
  const chapters = samples.filter((row, i) => row.phase && (!i || row.phase !== samples[i-1].phase));
  // Long, chattering labels are better inspected on the continuous timeline.
  if (chapters.length < 2 || chapters.length > 20) return null;
  return <div className="hand-trial-selector" aria-label="Recorded phases">{chapters.map((row,i)=><Button key={i} variant="ghost" onClick={()=>seek(row.t)}>{row.phase} · {row.t.toFixed(1)}s</Button>)}</div>;
}
