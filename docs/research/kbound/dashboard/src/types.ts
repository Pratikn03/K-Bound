export type StatusKind =
  | "verified"
  | "conditional"
  | "no_harm"
  | "pending"
  | "diagnostic"
  | "open"
  | "failed";

export interface Snapshot {
  meta?: {
    build_id?: string;
    paper?: string;
    paper_pages?: number;
  };
  research_status?: Record<string, string>;
  evidence_strip?: Record<string, { value?: string | number; sub?: string }>;
  regime_map?: Array<{
    id: string;
    title: string;
    action: string;
    status: StatusKind;
    examples: string;
    artifact?: string;
  }>;
  theory_ledger?: Array<{
    id: string;
    name: string;
    status: StatusKind;
    artifact?: string | null;
    implication: string;
    evidence: string;
  }>;
  headline_controlled?: PolicyRow[];
  evidence_board?: {
    controlled_wins?: PolicyRow[];
    helpful_dominated?: PolicyRow[];
    natural_shift_no_harm?: NaturalRow[];
    boundary_negative?: BoundaryRow[];
  };
  edge_validation?: EdgeValidation;
  safety?: {
    metrics?: Array<{ label: string; value: string | number; meaning: string }>;
    prose?: Record<string, string>;
  };
  reproduce?: {
    primary: string;
    gpu: string;
    validators: string;
    dashboard?: string;
    runtime_estimate: string;
    inputs?: string[];
    outputs?: string[];
  };
  provenance?: Record<string, string | null | undefined>;
}

export interface PolicyRow {
  name: string;
  status: StatusKind;
  artifact: string;
  framing: string;
  freeze?: number | null;
  adapt?: number | null;
  kga?: number | null;
  oracle?: number | null;
  regret_kga?: number | null;
  regret_adapt?: number | null;
  regret_freeze?: number | null;
  false_adapt?: number | null;
  beats_both_artifact?: boolean;
}

export interface NaturalRow {
  name: string;
  status: StatusKind;
  artifact: string;
  protocol?: string;
  regret_kga?: number | null;
  regret_adapt?: number | null;
  regret_freeze?: number | null;
  false_adapt?: number | null;
  framing: string;
}

export interface BoundaryRow {
  name: string;
  status: StatusKind;
  artifact: string;
  framing: string;
  freeze?: number | null;
  adapt?: number | null;
  kga?: number | null;
  oracle?: number | null;
  regret_kga?: number | null;
  abstention_rate?: number | null;
  false_adapt?: number | null;
  note?: string;
  pooled?: Record<string, unknown>;
}

export interface EdgeValidation {
  study_status: StatusKind;
  study_label: string;
  phases: Array<{
    id: string;
    label: string;
    status: StatusKind;
    detail: string;
    artifact?: string;
  }>;
  session_progress?: Array<{
    session: string;
    expected_clips: number;
    captured_clips: number;
    complete: boolean;
  }>;
  development_metrics?: {
    note: string;
    phone_a_balanced_acc?: number;
    phone_a_macro_f1?: number;
    kga_abstain_rate?: number;
    latency_ms_mean?: number;
    latency_ms_p95?: number;
  };
  unblock?: {
    all_pass: boolean;
    gate_thresholds: { balanced_acc: number; macro_f1: number };
    current: Record<string, boolean | number | string | null>;
    gaps: Array<{ check: string; passed: boolean; detail: string }>;
    commands: Record<string, string>;
  };
  protocol_hash?: string;
}

export interface RouteDef {
  path: string;
  id: string;
  label: string;
}
