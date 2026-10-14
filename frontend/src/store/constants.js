export const STAGE_ORDER = ["1", "2", "3", "4", "5", "6", "6a", "7", "8", "9"];

export const STAGE_ITEMS = [
  { id: "1", name: "Intent Ingestion", sub: "Parse concept, normalize inputs" },
  { id: "2", name: "Contextual Enhancement", sub: "Platform specs, trends, demographics" },
  { id: "3", name: "Narrative Sequencing", sub: "Shot-by-shot storyboard creation" },
  { id: "4", name: "Prompt Engineering", sub: "Cinematic prompts + camera choreography" },
  { id: "5", name: "Structural Validation", sub: "Schema check, API compatibility" },
  { id: "6", name: "Visual Synthesis", sub: "RunwayML gen4_image frames" },
  { id: "6a", name: "VLM Validation", sub: "GPT-4o Vision quality gate" },
  { id: "7", name: "Temporal Synthesis", sub: "Image-to-video clip generation" },
  { id: "8", name: "Non-Linear Stitching", sub: "FFmpeg concat → final MP4" },
  { id: "9", name: "Audio Production", sub: "Voiceover script + TTS mux" },
];

export const VID_COST = {
  gen3a_turbo: 5,
  gen4_turbo: 5,
  "gen4.5": 12,
  seedance2: 36,
  veo3: 40,
  "veo3.1": 30,
  "veo3.1_fast": 12,
};

export const IMG_COST = {
  gen4_image: 6,
  gen4_image_turbo: 2,
  "gemini_2.5_flash": 5,
};

let logSeq = 0;
export function nextLogId() {
  logSeq += 1;
  return logSeq;
}
