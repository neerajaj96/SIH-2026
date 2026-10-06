-- production_schema.sql
--
-- The demo console (netrasetu.html) uses the Claude artifact `db`
-- capability - a realtime JSON document store capped at roughly 5,000
-- documents per artifact. That's fine for a hackathon demo queue; it is
-- nowhere near "100,000+ patients annually" (the PS's own stated
-- target), which is more like 300+ screenings/day even spread evenly.
-- This is the schema an actual deployment migrates to. Written for
-- PostgreSQL (works with minor tweaks on most relational databases);
-- untested against a live database from this environment - review with
-- whoever owns real deployment infrastructure before running it there.
--
-- DESIGN NOTES:
--   - Normalized: patients, sites, and cases are separate tables, not
--     one denormalized blob per case (patients get screened more than
--     once; artifact db had no real way to express that relationship).
--   - Images are NOT stored as base64 in a row (a real image, not a
--     160x160 demo thumbnail, would bloat the database badly at scale) -
--     store them in object storage (S3-compatible) and keep only the
--     path/URL here.
--   - Indexes are chosen for the two access patterns the app actually
--     needs: "show me the pending queue" and "show me one patient's
--     history" - not a generic index-everything approach.
--   - `model_version` on each case ties back to the .meta.json sidecar
--     saveModelWithMetadata.m writes, so you can always answer "which
--     model version produced this grade" - the artifact-db demo had no
--     way to answer that at all.

CREATE TABLE sites (
    site_id         SERIAL PRIMARY KEY,
    site_name       TEXT NOT NULL,
    district        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE patients (
    patient_id      SERIAL PRIMARY KEY,
    site_id         INTEGER NOT NULL REFERENCES sites(site_id),
    external_ref    TEXT,               -- the PHC's own patient ID/card number, not this system's internal id
    date_of_birth   DATE,               -- nullable - rural registration often can't confirm this precisely
    sex             TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (site_id, external_ref)
);

CREATE TABLE reviewers (
    reviewer_id     SERIAL PRIMARY KEY,
    full_name       TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('ophthalmologist','technician','admin')),
    license_no      TEXT,               -- for ophthalmologists - real credential, not just a display name
    active          BOOLEAN NOT NULL DEFAULT true,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE cases (
    case_id             BIGSERIAL PRIMARY KEY,
    patient_id          INTEGER NOT NULL REFERENCES patients(patient_id),
    site_id             INTEGER NOT NULL REFERENCES sites(site_id),
    submitted_by        INTEGER REFERENCES reviewers(reviewer_id),  -- technician who captured it
    eye                 TEXT CHECK (eye IN ('OD','OS')),            -- right/left, standard ophthalmic notation - not tracked at all in the demo
    image_object_key    TEXT NOT NULL,   -- pointer into object storage, NOT the image itself
    submitted_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    focus_score          REAL,
    entropy_score        REAL,
    quality_passed        BOOLEAN NOT NULL,

    model_version         TEXT,            -- ties to a saveModelWithMetadata.m sidecar
    dl_grade               SMALLINT CHECK (dl_grade BETWEEN 0 AND 4),
    dl_confidence            REAL,
    rule_grade              SMALLINT NOT NULL CHECK (rule_grade BETWEEN 0 AND 4),
    rule_evidence            JSONB NOT NULL DEFAULT '[]',   -- array of evidence strings, same shape assignClinicalGrade.m already returns
    grades_disagree          BOOLEAN NOT NULL DEFAULT false,
    nv_flagged                BOOLEAN NOT NULL DEFAULT false,
    gradcam_on_disc            BOOLEAN NOT NULL DEFAULT false,
    gradcam_object_key          TEXT,          -- pointer to the overlay image, if rendered

    status                TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','reviewed','escalated')),
    reviewed_by             INTEGER REFERENCES reviewers(reviewer_id),
    reviewer_grade            SMALLINT CHECK (reviewer_grade BETWEEN 0 AND 4),
    reviewer_notes              TEXT,
    reviewed_at                 TIMESTAMPTZ
);

-- "show me the pending queue, oldest first" - the review view's main query
CREATE INDEX idx_cases_pending_queue ON cases (status, submitted_at) WHERE status = 'pending';

-- "show me this patient's screening history" - the clinical-continuity query the demo has no equivalent of at all
CREATE INDEX idx_cases_patient_history ON cases (patient_id, submitted_at DESC);

-- "how many referable cases came out of site X in date range Y" - the admin/capacity view's query
CREATE INDEX idx_cases_site_referable ON cases (site_id, submitted_at) WHERE rule_grade >= 2 OR dl_grade >= 2;

-- Row-level security sketch (Postgres RLS) - the demo's db capability
-- enforces access at the ARTIFACT level (org members read/write, others
-- view-only or nothing); a real deployment needs it at the ROW level -
-- a technician at Site A should not see Site B's patient data by
-- default. This is a STARTING POINT, not a complete policy - a real
-- clinical deployment needs a security review beyond what's sketched here.
ALTER TABLE cases ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY site_isolation ON cases
--     USING (site_id = current_setting('app.current_site_id')::INTEGER);
