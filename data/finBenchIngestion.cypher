
CREATE INDEX IF NOT EXISTS FOR (a:Account) ON (a.accountId);
CREATE INDEX IF NOT EXISTS FOR (l:Loan) ON (l.loanId);
CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.companyId);
CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.personId);
CREATE INDEX IF NOT EXISTS FOR (m:Medium) ON (m.mediumId);


CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/Account.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "CREATE (:Account {
    accountId: toInteger(row.accountId),
    createTime: datetime(replace(row.createTime, ' ', 'T')),
    isBlocked: toBoolean(row.isBlocked),
    accountType: toString(row.accountType),
    nickname: toString(row.nickname),
    phonenum: toString(row.phonenum),
    email: toString(row.email),
    freqLoginType: toString(row.freqLoginType),
    lastLoginTime: datetime({epochMillis: toInteger(row.lastLoginTime)}),
    accountLevel: toString(row.accountLevel)
  })",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/Company.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "CREATE (:Company {
    companyId: toInteger(row.companyId),
    companyName: toString(row.companyName),
    isBlocked: toBoolean(row.isBlocked),
    createTime: datetime(replace(row.createTime, ' ', 'T')),
    country: toString(row.country),
    city: toString(row.city),
    business: toString(row.business),
    description: toString(row.description),
    url: toString(row.url)
  })",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/Loan.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "CREATE (:Loan {
    loanId: toInteger(row.loanId),
    loanAmount: toFloat(row.loanAmount),
    balance: toFloat(row.balance),
    createTime: datetime(replace(row.createTime, ' ', 'T')),
    loanUsage: toString(row.loanUsage),
    interestRate: toFloat(row.interestRate),
    flow_rate: 'TS_flow_rate'
  })",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/Medium.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "CREATE (:Medium {
    mediumId: toInteger(row.mediumId),
    mediumType: toString(row.mediumType),
    isBlocked: toBoolean(row.isBlocked),
    createTime: datetime(replace(row.createTime, ' ', 'T')),
    lastLoginTime: datetime({epochMillis: toInteger(row.lastLoginTime)}),
    riskLevel: toString(row.riskLevel)
  })",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/Person.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "CREATE (:Person {
    personId: toInteger(row.personId),
    personName: toString(row.personName),
    isBlocked: toBoolean(row.isBlocked),
    createTime: datetime(replace(row.createTime, ' ', 'T')),
    gender: toString(row.gender),
    birthday: datetime(replace(row.birthday, ' ', 'T')),
    country: toString(row.country),
    city: toString(row.city)
  })",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/AccountRepayLoan.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "MATCH (a:Account {accountId: toInteger(row.accountId)}), (l:Loan {loanId: toInteger(row.loanId)})
   CREATE (a)-[:Repay {amount: toFloat(row.amount), createTime: datetime(replace(row.createTime, ' ', 'T'))}]->(l)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/LoanDepositAccount.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "MATCH (a:Account {accountId: toInteger(row.accountId)}), (l:Loan {loanId: toInteger(row.loanId)})
   CREATE (l)-[:Deposit {amount: toFloat(row.amount), createTime: datetime(replace(row.createTime, ' ', 'T'))}]->(a)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/AccountTransferAccount.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "MATCH (a:Account {accountId: toInteger(row.fromId)}), (b:Account {accountId: toInteger(row.toId)})
   CREATE (a)-[:Transfer {
     amount: toFloat(row.amount),
     createTime: datetime(replace(row.createTime, ' ', 'T')),
     orderNum: toInteger(row.orderNum),
     comment: toString(row.comment),
     payType: toString(row.payType),
     goodsType: toString(row.goodsType)
   }]->(b)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/AccountWithdrawAccount.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "MATCH (a:Account {accountId: toInteger(row.fromId)}), (b:Account {accountId: toInteger(row.toId)})
   CREATE (a)-[:Withdraw {
     amount: toFloat(row.amount),
     createTime: datetime(replace(row.createTime, ' ', 'T'))
   }]->(b)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/CompanyApplyLoan.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "MATCH (c:Company {companyId: toInteger(row.companyId)}), (l:Loan {loanId: toInteger(row.loanId)})
   CREATE (c)-[:Apply {
     createTime: datetime(replace(row.createTime, ' ', 'T')),
     org: toString(row.org)
   }]->(l)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/CompanyGuaranteeCompany.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "MATCH (a:Company {companyId: toInteger(row.fromId)}), (b:Company {companyId: toInteger(row.toId)})
   CREATE (a)-[:Guarantee {
     createTime: datetime(replace(row.createTime, ' ', 'T')),
     relation: toString(row.relation)
   }]->(b)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/CompanyInvestCompany.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "MATCH (a:Company {companyId: toInteger(row.investorId)}), (b:Company {companyId: toInteger(row.companyId)})
   CREATE (a)-[:Invest {
     createTime: datetime(replace(row.createTime, ' ', 'T')),
     ratio: toFloat(row.ratio)
   }]->(b)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/CompanyOwnAccount.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "MATCH (c:Company {companyId: toInteger(row.companyId)}), (a:Account {accountId: toInteger(row.accountId)})
   CREATE (c)-[:Own {
     createTime: datetime(replace(row.createTime, ' ', 'T'))
   }]->(a)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/MediumSignInAccount.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "MATCH (m:Medium {mediumId: toInteger(row.mediumId)}), (a:Account {accountId: toInteger(row.accountId)})
   CREATE (m)-[:SignIn {
     createTime: datetime(replace(row.createTime, ' ', 'T')),
     location: toString(row.location)
   }]->(a)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/PersonApplyLoan.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "MATCH (p:Person {personId: toInteger(row.personId)}), (l:Loan {loanId: toInteger(row.loanId)})
   CREATE (p)-[:Apply {
     createTime: datetime(replace(row.createTime, ' ', 'T')),
     org: toString(row.org)
   }]->(l)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/PersonGuaranteePerson.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "MATCH (a:Person {personId: toInteger(row.fromId)}), (b:Person {personId: toInteger(row.toId)})
   CREATE (a)-[:Guarantee {
     createTime: datetime(replace(row.createTime, ' ', 'T')),
     relation: toString(row.relation)
   }]->(b)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/PersonInvestCompany.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "MATCH (a:Person {personId: toInteger(row.investorId)}), (b:Company {companyId: toInteger(row.companyId)})
   CREATE (a)-[:Invest {
     createTime: datetime(replace(row.createTime, ' ', 'T')),
     ratio: toFloat(row.ratio)
   }]->(b)",
  {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///finbench/PersonOwnAccount.csv' AS row FIELDTERMINATOR '|' RETURN row",
  "MATCH (p:Person {personId: toInteger(row.personId)}), (a:Account {accountId: toInteger(row.accountId)})
   CREATE (p)-[:Own {
     createTime: datetime(replace(row.createTime, ' ', 'T'))
   }]->(a)",
  {batchSize: 1000, parallel: false}
);



MATCH ()-[r]->() set r.toDelete=False; 


match (a) set a.isBlocked_GT = a.isBlocked;

UNWIND range(1, 1000) AS id
WITH id, apoc.create.uuid() AS uuid
MERGE (p:Person {id: id})
SET p.isViolation = true,
    p.violationId = coalesce(p.violationId, []) + uuid
MERGE (p)-[r:Guarantee]->(p)
with r
SET r.isViolation = true,
    r.violationId = coalesce(r.violationId, []) + uuid,
    r.toDelete = true;
CREATE (v:Violation {
      violationId: uuid,
      repairs: '[MATCH ()-[gua:Guarantee]-() WHERE gua.elementId=$r SET gua.deleted=true]',
      order: [0],
      
  })

  CALL apoc.periodic.iterate(
  "
  MATCH (a:Account)-[r:Transfer]->(a1:Account)
  WHERE a.isBlocked = true
  RETURN DISTINCT a, a1, r
  ",
  "
  WITH a, a1, r, apoc.create.uuid() AS uuid, rand() AS randomValue
  SET a.isViolation = true,
      a1.isViolation = true,
      r.isViolation = true,
      a.violationId = coalesce(a.violationId, []) + uuid,
      a1.violationId = coalesce(a1.violationId, []) + uuid,
      r.violationId = coalesce(r.violationId, []) + uuid
  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    SET a.isBlocked_GT = false
     CREATE (v:Violation {
      violationId: uuid,
      repairs: '[MATCH ()-[tr:Transfer]-() WHERE tr.elementId=$r SET tr.deleted=true,MATCH (acc:Account) WHERE acc.elementId=$a SET acc.isBlocked=False,MATCH (acc:Account)-[tr:Transfer]-() WHERE acc.elementId=$a AND tr.elementId=$r SET tr.deleted=true, a.isBlocked=False]',
      order: [1,2,0],
  })

  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
    SET r.toDelete = true
    CREATE (v:Violation {
      violationId: uuid,
      repairs: '[MATCH ()-[tr:Transfer]-() WHERE tr.elementId=$r SET tr.deleted=true,MATCH (acc:Account) WHERE acc.elementId=$a SET acc.isBlocked=False,MATCH (acc:Account)-[tr:Transfer]-() WHERE acc.elementId=$a AND tr.elementId=$r SET tr.deleted=true, a.isBlocked=False]',
      order: [0,2,1],
  })
  )
  ",
  {
    batchSize: 100,
    parallel: false
  }
);


MATCH (a:Account)
WITH a, toInteger(rand() * (1000000000 - 1000)) + 1000 AS newBalance
SET a.balance = newBalance;

MATCH (a:Account)
SET a.balance_GT =a.balance;

CALL apoc.periodic.iterate(
  "
  MATCH (guarantor:Person)-[s:Guarantee]->(guaranteed:Person)-[r:Own]->(account:Account)
  WHERE guarantor <> guaranteed AND account.balance < 500000000
  RETURN DISTINCT guarantor, guaranteed, s, r, account
  ",
  "
  WITH guarantor, guaranteed, s, r, account, apoc.create.uuid() AS uuid, rand() AS randomValue

  SET guarantor.isViolation = true,
      guaranteed.isViolation = true,
      s.isViolation = true,
      r.isViolation = true,
      account.isViolation = true,
      guarantor.violationId = coalesce(guarantor.violationId, []) + uuid,
      guaranteed.violationId = coalesce(guaranteed.violationId, []) + uuid,
      s.violationId = coalesce(s.violationId, []) + uuid,
      r.violationId = coalesce(r.violationId, []) + uuid,
      account.violationId = coalesce(account.violationId, []) + uuid

  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    SET s.toDelete = True
  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
    SET account.balance_GT = 500000001 + toInteger(rand() * (1000000000 - 500000001))
  )
  ",
  {
    batchSize: 100,
    parallel: false
  }
);


MATCH (a:Loan)
SET a.interestRate_GT =a.interestRate;


CALL apoc.periodic.iterate(
  "
  MATCH (c:Company)-[r:Apply]->(l:Loan)
  WHERE l.interestRate < 0.02
  RETURN DISTINCT c, r, l
  ",
  "
  WITH c, r, l, apoc.create.uuid() AS uuid, rand() AS randomValue

  SET c.isViolation = true,
      r.isViolation = true,
      l.isViolation = true,

      c.violationId = coalesce(c.violationId, []) + uuid,
      r.violationId = coalesce(r.violationId, []) + uuid,
      l.violationId = coalesce(l.violationId, []) + uuid

  FOREACH (_ IN CASE WHEN randomValue < 0.5 THEN [1] ELSE [] END |
    SET r.toDelete = true
  )
  FOREACH (_ IN CASE WHEN randomValue >= 0.5 THEN [1] ELSE [] END |
    SET l.interestRate_GT = 0.02 + rand() * (0.1 - 0.02)
  )
  ",
  {
    batchSize: 100,
    parallel: false
  }
);


