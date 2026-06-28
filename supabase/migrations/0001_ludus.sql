-- REFERENCE SCHEMA ONLY — NON-AUTHORITATIVE.
-- InsForge does not apply raw SQL; tables are created via its admin API
-- (POST /api/database/tables). The authoritative, runnable mechanism is
-- `scripts/insforge_setup.py`, which creates ludus_episodes + ludus_steps.
-- This file documents the equivalent relational schema for readers using a
-- plain Postgres backend. (ludus_rules below is illustrative — the live code
-- stores learned rules inline on ludus_episodes.rules, so the setup script
-- does not create a separate ludus_rules table.)
create table if not exists ludus_episodes (
  episode_id text primary key,
  game text not null,
  mode text not null,
  steps int not null,
  legal_action_rate double precision not null,
  final_metrics jsonb not null,
  rules jsonb not null,
  created_at timestamptz default now()
);
create table if not exists ludus_steps (
  id bigserial primary key,
  episode_id text references ludus_episodes(episode_id),
  step_index int not null,
  mode text not null,
  game text not null,
  action text not null,
  expected_result text,
  primary_metric text,
  primary_delta double precision,
  improved boolean,
  metric_delta jsonb,
  rule_added text,
  screenshot_ref text,
  confidence double precision,
  created_at timestamptz default now()
);
create table if not exists ludus_rules (
  id bigserial primary key,
  episode_id text references ludus_episodes(episode_id),
  rule text not null,
  created_at timestamptz default now()
);
