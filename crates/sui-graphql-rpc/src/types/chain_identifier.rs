// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    data::{Db, DbConnection, QueryExecutor},
    error::Error,
};
use async_graphql::*;
use diesel::{ExpressionMethods, QueryDsl};
use once_cell::sync::OnceCell;
use sui_indexer::schema::checkpoints;
use sui_protocol_config::Chain;
use sui_types::{
    digests::ChainIdentifier as NativeChainIdentifier, messages_checkpoint::CheckpointDigest,
};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ChainId {
    pub(crate) identifier: NativeChainIdentifier,
    pub(crate) chain: Chain,
}

static ACTIVE_CHAIN_ID: OnceCell<ChainId> = OnceCell::new();

pub(crate) struct ChainIdentifier;

impl ChainId {
    pub(crate) fn chain(&self) -> &Chain {
        &self.chain
    }

    pub(crate) fn identifier(&self) -> &NativeChainIdentifier {
        &self.identifier
    }
}

impl ChainIdentifier {
    /// Get the chain_id. Saves in cache (once) throughout the service lifecycle.
    /// Gets initialized from the DB if not found in cache.
    pub(crate) async fn get_chain_id(db: &Db) -> Option<ChainId> {
        if let Some(chain_id) = ACTIVE_CHAIN_ID.get() {
            return Some(*chain_id);
        };

        let queried_id = Self::query(db).await.ok()?;

        let chain_id = ChainId {
            identifier: queried_id,
            chain: queried_id.chain(),
        };

        ACTIVE_CHAIN_ID.set(chain_id).ok()?;

        Some(chain_id)
    }

    /// Query the Chain Identifier from the DB.
    async fn query(db: &Db) -> Result<NativeChainIdentifier, Error> {
        use checkpoints::dsl;

        let digest_bytes = db
            .execute(move |conn| {
                conn.first(move || {
                    dsl::checkpoints
                        .select(dsl::checkpoint_digest)
                        .order_by(dsl::sequence_number.asc())
                })
            })
            .await
            .map_err(|e| Error::Internal(format!("Failed to fetch genesis digest: {e}")))?;

        Self::from_bytes(digest_bytes)
    }

    /// Treat `bytes` as a checkpoint digest and extract a chain identifier from it.
    pub(crate) fn from_bytes(bytes: Vec<u8>) -> Result<NativeChainIdentifier, Error> {
        let genesis_digest = CheckpointDigest::try_from(bytes)
            .map_err(|e| Error::Internal(format!("Failed to deserialize genesis digest: {e}")))?;
        Ok(NativeChainIdentifier::from(genesis_digest))
    }
}
