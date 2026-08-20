/**
 * TypeScript Data Transfer Objects (DTO) matching the FastAPI Pydantic v2 schemas.
 */

export interface ActionDTO {
  action_type: string;
  card_index?: number | null;
  route_id?: string | null;
  card_color?: string | null;
  ticket_ids?: string[] | null;
}

export interface PlayerStateDTO {
  player_id: string;
  name: string;
  score: number;
  trains_remaining: number;
  cards_in_hand: Record<string, number>;
  tickets: Array<{
    id: string;
    city_a: string;
    city_b: string;
    points: number;
    completed: boolean;
  }>;
  claimed_route_ids: string[];
  color: string;
}

export interface GameStateDTO {
  session_id: string;
  turn_number: number;
  current_player_index: number;
  map_name: string;
  players: PlayerStateDTO[];
  visible_cards: string[];
  deck_size: number;
  discard_pile_size: number;
  tickets_deck_size: number;
  claimed_routes: Record<string, string>; // route_id -> player_id
  valid_actions: ActionDTO[];
  action_mask: boolean[];
  is_game_over: boolean;
  winner_id?: string | null;
  last_action?: ActionDTO | null;
  last_reward?: number | null;
}

export interface GameSessionCreateRequest {
  map_name?: string;
  player_types?: string[];
  seed?: number;
  model_checkpoint?: string | null;
}

export interface LayerActivationDTO {
  layer_name: string;
  shape: number[];
  mean: number;
  std: number;
  min: number;
  max: number;
  values: number[];
}

export interface BrainInspectionDTO {
  model_type: 'dqn' | 'ppo';
  observation_vector: number[];
  action_mask: boolean[];
  layer_activations: LayerActivationDTO[];
  raw_logits_or_q: number[];
  masked_logits_or_q: number[];
  action_probabilities: number[];
  estimated_value?: number | null;
  greedy_action_index: number;
  action_labels: string[];
}

export interface TrainingStartRequest {
  config_name: string;
  override_timesteps?: number | null;
  seed?: number;
  opponent_type?: 'random' | 'greedy' | 'strategic';
}

export interface TrainingStatusDTO {
  is_training: boolean;
  experiment_id?: string | null;
  algorithm?: string | null;
  current_step: number;
  total_timesteps: number;
  episodes: number;
  mean_reward: number;
}

export interface TelemetryEventDTO {
  type: 'training_started' | 'training_step' | 'checkpoint_saved' | 'training_finished' | 'error';
  experiment_id: string;
  step: number;
  episode: number;
  reward: number;
  mean_reward: number;
  policy_loss?: number | null;
  value_loss?: number | null;
  entropy?: number | null;
  approx_kl?: number | null;
  win_rate?: number | null;
  fps?: number | null;
}

export interface ReplayFrameDTO {
  step_index: number;
  turn_number: number;
  player_index: number;
  action: ActionDTO;
  reward: number;
  state_snapshot: Record<string, any>;
  observation?: number[] | null;
  action_mask?: boolean[] | null;
  action_probabilities?: number[] | null;
}

export interface ReplayDetailDTO {
  replay_id: string;
  map_name: string;
  seed: number;
  date: string;
  player_names: string[];
  total_steps: number;
  winner_index: number;
  final_scores: number[];
  frames: ReplayFrameDTO[];
}

export interface ExperimentRecordDTO {
  experiment_id: string;
  name: string;
  algorithm: string;
  seed: number;
  timestamp: string;
  metrics: Record<string, number>;
  config: Record<string, any>;
}

export interface CheckpointDTO {
  checkpoint_id: string;
  name: string;
  algorithm: string;
  path: string;
  size_mb: number;
  modified_at: string;
  total_timesteps?: number | null;
}

export interface TournamentParticipantOptionDTO {
  id: string;
  name: string;
  category: 'baseline' | 'checkpoint';
  algorithm: string;
  checkpoint_path?: string | null;
  description?: string | null;
}

export interface TournamentAgentDTO {
  agent_id: string;
  name: string;
  elo: number;
  win_rate: number;
  wins: number;
  losses: number;
  draws: number;
  avg_score: number;
  total_games: number;
}

export interface TournamentMatchupDTO {
  agent_a: string;
  agent_b: string;
  wins_a: number;
  wins_b: number;
  draws: number;
  win_rate_a: number;
  avg_score_a: number;
  avg_score_b: number;
  games_played: number;
}

export interface TournamentLeaderboardDTO {
  leaderboard: TournamentAgentDTO[];
  matchups: TournamentMatchupDTO[];
  total_games: number;
  updated_at: string;
  map_name?: string;
  available_participants?: TournamentParticipantOptionDTO[];
}

export interface TournamentRunRequest {
  participant_ids?: string[];
  games_per_pair?: number;
  map_name?: 'usa' | 'mini';
  seed?: number;
}

export interface ReportItemDTO {
  id: string;
  name: string;
  filename: string;
  file_type: 'markdown' | 'json' | 'text';
  size_kb: number;
  modified_at: string;
  phase?: string | null;
}

export interface ReportDetailDTO {
  id: string;
  name: string;
  filename: string;
  file_type: 'markdown' | 'json' | 'text';
  raw_content: string;
  json_data?: Record<string, any> | null;
  size_kb: number;
  modified_at: string;
}

