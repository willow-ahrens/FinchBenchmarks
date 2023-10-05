//
// Created by Daniel Donenfeld on 6/10/21.
//

#include "taco/lower/mode_format_rle.h"

#include "ir/ir_generators.h"
#include "taco/ir/simplify.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

constexpr const int headerSize = 2;
taco::Datatype headerType = headerSize == 1 ? 
                              taco::UInt8 : 
                            headerSize == 2 ? 
                              taco::UInt16 : 
                            headerSize == 4 ? 
                              taco::UInt32 : 
                              taco::UInt64;

using header_type = typename std::conditional<headerSize == 1, 
                                                uint8_t, 
                             std::conditional<headerSize == 2, 
                                                uint16_t, 
                             std::conditional<headerSize == 4, 
                                                uint32_t, 
                                                uint64_t>::type>::type>::type;

const int header_max = std::numeric_limits<header_type>::max() + 1;
const int header_mid = std::numeric_limits<header_type>::max()/2;


namespace taco {

    RLEModeFormat::RLEModeFormat() :
            RLEModeFormat(false, true, true, false) {
    }

    RLEModeFormat::RLEModeFormat(bool isFull, bool isOrdered,
                                   bool isUnique, bool isZeroless,
                                   long long allocSize) :
            ModeFormatImpl("rle", isFull, isOrdered, isUnique, false, true,
                           isZeroless, false, true, false, false,
                           true, false, false, false, true,
                           true, taco_positer_kind::BYTE,
                           false),
            allocSize(allocSize) {
    }

    ModeFormat RLEModeFormat::copy(
            vector<ModeFormat::Property> properties) const {
      bool isFull = this->isFull;
      bool isOrdered = this->isOrdered;
      bool isUnique = this->isUnique;
      bool isZeroless = this->isZeroless;
      for (const auto property : properties) {
        switch (property) {
          case ModeFormat::FULL:
            isFull = true;
            break;
          case ModeFormat::NOT_FULL:
            isFull = false;
            break;
          case ModeFormat::ORDERED:
            isOrdered = true;
            break;
          case ModeFormat::NOT_ORDERED:
            isOrdered = false;
            break;
          case ModeFormat::UNIQUE:
            isUnique = true;
            break;
          case ModeFormat::NOT_UNIQUE:
            isUnique = false;
            break;
          case ModeFormat::ZEROLESS:
            isZeroless = true;
            break;
          case ModeFormat::NOT_ZEROLESS:
            isZeroless = false;
            break;
          default:
            break;
        }
      }
      const auto compressedVariant =
              std::make_shared<RLEModeFormat>(isFull, isOrdered, isUnique,
                                                        isZeroless);
      return ModeFormat(compressedVariant);
    }

    Expr ternaryOp(const Expr& c, const Expr& a, const Expr& b);


  ModeFunction RLEModeFormat::posIterBounds(Expr parentPos,
                                                        Mode mode) const {
      Expr pbegin = Load::make(getPosArray(mode.getModePack()), parentPos);
      Expr pend = Load::make(getPosArray(mode.getModePack()),
                             ir::Add::make(parentPos, 1));

      Expr runVar = getVar("_run", mode);
      Expr foundCount = getFoundCountVar(mode);
      Expr posAccessCoord = getPosCoordVar(mode);
      Expr coordVar = getCoordVar(mode);

      Stmt runDecl = VarDecl::make(runVar, 0);
      Stmt foundDecl = VarDecl::make(foundCount, 0);
      Stmt coordDecl = VarDecl::make(posAccessCoord, coordVar);


      Stmt blk =  Block::make(VarDecl::make(getCoordVar(mode), 0),
                              runDecl, foundDecl, coordDecl);
      return ModeFunction(blk, {pbegin, pend});
    }

    ModeFunction RLEModeFormat::coordBounds(Expr parentPos,
                                                      Mode mode) const {
      taco_not_supported_yet;
      return ModeFunction();
    }

    Expr LeftShift_make(Expr lhs, Expr rhs);

    Expr RightShift_make(Expr lhs, Expr rhs);

    Expr loadByType(ir::Expr values, ir::Expr pos, Datatype type);

    Stmt storeByType(ir::Expr storeVal, ir::Expr values, ir::Expr pos, Datatype type);

    Expr msbCheck(ir::Expr values, ir::Expr pos, int bitnum, int value);

    template <class T>
    Expr setHighBit(const ir::Expr& value, bool bit){
      Datatype dtype = type<T>();
      T mask = 1 << (dtype.getNumBits()-1);
      if(bit){
        return ir::BitOr::make(value, mask);
      } else {
        mask = ~mask;
        return ir::BitAnd::make(value, mask);
      }
    }

    template <class T>
    Expr clearHighBit(const ir::Expr& value) {
      return setHighBit<T>(value, false);
    }

    template <class T>
    Expr setHighBit(const ir::Expr& value) {
      return setHighBit<T>(value, true);
    }

    ModeFunction RLEModeFormat::posIterAccess(std::vector<ir::Expr> pos_,
                                               std::vector<ir::Expr> coords,
                                               ir::Expr values, Datatype type,
                                               Mode mode) const {
      taco_iassert(mode.getPackLocation() == 0);
      ir::Expr pos = pos_.back();

      // Header byte	Data following the header byte
      // 0 to 127	    (1 + n) literal bytes of data
      // −1 to −128	    One byte of data, repeated (1 − n) times in the decompressed output


      // 0 XXXXXXX XXXXXXXX     -> read X number of bytes
      // 1 XXXXXXX XXXXXXXX Y Y -> X is the run length, Y is the distance

      // Code for finding raw values
      Expr posAccessCoord = getPosCoordVar(mode);
      Expr coordVar = getCoordVar(mode);
      Expr foundCount = getFoundCountVar(mode);

      Stmt coordDecl = Assign::make(posAccessCoord, coordVar);

      // Expr headerByte = Load::make(values, pos); //loadByType(values, pos, UInt8);
      Expr headerByte = loadByType(values, pos, headerType);

      // Expr runCheck = Gt::make(headerByte, header_mid+1);
      Expr runCheck = msbCheck(values, pos, headerSize*8 - 1, 1);

      Expr runVar = getVar("_run", mode);
      Stmt runBody = Block::make(
              Assign::make(foundCount, 0),
              // Assign::make(runVar, ir::Sub::make(header_max, headerByte)),
              Assign::make(runVar, clearHighBit<header_type>(headerByte)),
              addAssign(pos,ir::Add::make(headerSize, type.getNumBytes())), addAssign(getCoordVar(mode), runVar), Assign::make(posAccessCoord, coordVar)
      );

      Stmt litBody = Block::make(
              // Assign::make(foundCount, ir::Add::make(headerByte, 1)),
              Assign::make(foundCount, headerByte),
              addAssign(pos, headerSize),
              addAssign(coordVar, foundCount));

      // Code for updating fill region
      auto fillRegion = Load::make(values, ir::Sub::make(pos, type.getNumBytes()), true);
      if (type.getKind() != Datatype::UInt8){
        fillRegion = ir::Cast::make(fillRegion, type, true);
      }

      Stmt ifStmt = IfThenElse::make(runCheck, runBody, litBody);

      Stmt blk = Block::make(coordDecl, ifStmt);
      return ModeFunction(blk, {posAccessCoord, foundCount, fillRegion, 1, ir::Neg::make(ir::Cast::make(foundCount, Bool))});
    }

    Expr varFromCast(Expr var);

    Stmt RLEModeFormat::getAppendCoord(Expr p, Expr i, Expr values, Expr valuesCap, Datatype type, Mode mode) const {
       taco_iassert(mode.getPackLocation() == 0);
       auto countPos = getCurrCountVar(mode);
       auto countValVar = getCurrCountValVar(mode);
       auto isFilling = getFillingBoolVar(mode);

       auto addr = Load::make(values, p, true);
       auto currValue = Var::make("curr_value", type);
       auto currValueDecl = VarDecl::make(currValue, Load::make(ir::Cast::make(addr, type, true)));

       auto storeCountValVar = Block::make(storeByType(countValVar, values, countPos, UInt16));
       auto setupFilling = Block::make(
               IfThenElse::make(ir::And::make(isFilling, Gte::make(countValVar,0)), storeCountValVar),
               Assign::make(isFilling, true), Assign::make(countPos, p),
               doubleSizeIfFull(varFromCast(values), valuesCap, ir::Add::make(p,1)),
               Assign::make(countValVar, 0),  //storeByType(1, values, p, UInt16),
               storeByType(currValue, values, ir::Add::make(p,1), type),
               addAssign(p,1));

       auto cond = ir::And::make(isFilling, Lt::make(countValVar, 127));

       return Block::make(currValueDecl, IfThenElse::make(cond, Block::make(addAssign(countValVar, 1)), setupFilling));
    }

    Stmt RLEModeFormat::getAppendEdges(Expr pPrev, Expr pBegin, Expr pEnd,
                                              Mode mode) const {
      Expr posArray = getPosArray(mode.getModePack());
      ModeFormat parentModeType = mode.getParentModeType();
      Expr edges = (!parentModeType.defined() || parentModeType.hasAppend())
                   ? pEnd : ir::Sub::make(pEnd, pBegin);
      return Store::make(posArray, ir::Add::make(pPrev, 1), edges);
    }

    Expr RLEModeFormat::getSize(ir::Expr szPrev, Mode mode) const {
      return Load::make(getPosArray(mode.getModePack()), szPrev);
    }

    Stmt RLEModeFormat::getAppendInitEdges(Expr pPrevBegin,
                                                  Expr pPrevEnd, Mode mode) const {
      if (isa<ir::Literal>(pPrevBegin)) {
        taco_iassert(to<ir::Literal>(pPrevBegin)->equalsScalar(0));
        return Stmt();
      }

      Expr posArray = getPosArray(mode.getModePack());
      Expr posCapacity = getPosCapacity(mode);
      ModeFormat parentModeType = mode.getParentModeType();
      if (!parentModeType.defined() || parentModeType.hasAppend()) {
        return doubleSizeIfFull(posArray, posCapacity, pPrevEnd);
      }

      Expr pVar = Var::make("p" + mode.getName(), Int());
      Expr lb = ir::Add::make(pPrevBegin, 1);
      Expr ub = ir::Add::make(pPrevEnd, 1);
      Stmt initPos = For::make(pVar, lb, ub, 1, Store::make(posArray, pVar, 0));
      Stmt maybeResizePos = atLeastDoubleSizeIfFull(posArray, posCapacity, pPrevEnd);
      return Block::make({maybeResizePos, initPos});
    }

    Stmt RLEModeFormat::getAppendInitLevel(Expr szPrev, Expr sz,
                                                  Mode mode) const {
      const bool szPrevIsZero = isa<ir::Literal>(szPrev) &&
                                to<ir::Literal>(szPrev)->equalsScalar(0);

      Expr defaultCapacity = ir::Literal::make(allocSize, Datatype::Int32);
      Expr posArray = getPosArray(mode.getModePack());
      Expr initCapacity = szPrevIsZero ? defaultCapacity : ir::Add::make(szPrev, 1);
      Expr posCapacity = initCapacity;

      std::vector<Stmt> initStmts;
      if (szPrevIsZero) {
        posCapacity = getPosCapacity(mode);
        initStmts.push_back(VarDecl::make(posCapacity, initCapacity));
      }
      initStmts.push_back(Allocate::make(posArray, posCapacity));
      initStmts.push_back(Store::make(posArray, 0, 0));

      if (mode.getParentModeType().defined() &&
          !mode.getParentModeType().hasAppend() && !szPrevIsZero) {
        Expr pVar = Var::make("p" + mode.getName(), Int());
        Stmt storePos = Store::make(posArray, pVar, 0);
        initStmts.push_back(For::make(pVar, 1, initCapacity, 1, storePos));
      }

      auto countPos = getCurrCountVar(mode);
      initStmts.push_back(VarDecl::make(countPos, 0));

      auto countVal = getCurrCountValVar(mode);
      initStmts.push_back(VarDecl::make(countVal, 0));

      auto isFilling = getFillingBoolVar(mode);
      initStmts.push_back(VarDecl::make(isFilling, false));

      return Block::make(initStmts);
    }

    Stmt RLEModeFormat::getAppendFinalizeLevel(Expr szPrev,  Expr sz, Expr values,
                                                Mode mode) const {
      ModeFormat parentModeType = mode.getParentModeType();

      vector<Stmt> stmts;

      auto countPos = getCurrCountVar(mode);
      auto countValVar = getCurrCountValVar(mode);
      auto isFilling = getFillingBoolVar(mode);
      auto storeCountValVar = Block::make(storeByType(countValVar, values, countPos, Int8));
      stmts.push_back(IfThenElse::make(ir::And::make(isFilling, Gte::make(countValVar,0)), storeCountValVar));

      if ((isa<ir::Literal>(szPrev) && to<ir::Literal>(szPrev)->equalsScalar(1)) ||
          !parentModeType.defined() || parentModeType.hasAppend()) {
        return stmts.empty() ? Stmt() : Block::make(stmts);
      }

      Expr csVar = Var::make("cs" + mode.getName(), Int());
      Stmt initCs = VarDecl::make(csVar, 0);
      stmts.push_back(initCs);

      Expr pVar = Var::make("p" + mode.getName(), Int());
      Expr loadPos = Load::make(getPosArray(mode.getModePack()), pVar);
      Stmt incCs = Assign::make(csVar, ir::Add::make(csVar, loadPos));
      Stmt updatePos = Store::make(getPosArray(mode.getModePack()), pVar, csVar);
      Stmt body = Block::make({incCs, updatePos});
      Stmt finalizeLoop = For::make(pVar, 1, ir::Add::make(szPrev, 1), 1, body);
      stmts.push_back(finalizeLoop);

      return Block::make(stmts);
    }

    Stmt
    RLEModeFormat::getFillRegionAppend(ir::Expr p, ir::Expr i,
                                        ir::Expr start, ir::Expr length,
                                        ir::Expr run, ir::Expr values, ir::Expr valuesCap,
                                        Datatype type, Mode mode) const {
      // 1 XXXXXXX XXXXXXXX Y Y -> X is the run length, Y is the distance
      // TODO: This code is incorrect
      taco_iassert(isa<ir::Literal>(run) && to<ir::Literal>(run)->getIntValue() == 1);

      auto countPos = getCurrCountVar(mode);
      auto countValVar = getCurrCountValVar(mode);
      auto isFilling = getFillingBoolVar(mode);

      auto storeCountValVar = Block::make(storeByType(countValVar, values, countPos, Int8));
      auto setCount = IfThenElse::make(ir::And::make(isFilling, Gte::make(countValVar,0)), storeCountValVar);

      auto setFilling = Assign::make(getFillingBoolVar(mode), false);
      auto resizeVals = doubleSizeIfFull(varFromCast(values), valuesCap, ir::Add::make(p,4));
      Expr distValue = ir::Div::make(ir::Sub::make(p, start), type.getNumBytes());
      auto storeDist = storeByType(distValue, values, ir::Add::make(p,2), UInt16);
      auto storeRun = storeByType(setHighBit<uint16_t>(run,true), values, p, UInt16);

      return Block::make(setCount, setFilling, resizeVals, storeDist, storeRun, addAssign(p,4));
    }

    vector<Expr> RLEModeFormat::getArrays(Expr tensor, int mode,
                                                    int level) const {
      std::string arraysName = util::toString(tensor) + std::to_string(level);
      return {GetProperty::make(tensor, TensorProperty::Indices,
                                level - 1, 0, arraysName + "_pos")};

    }

    Expr RLEModeFormat::getPosArray(ModePack pack) const {
      return pack.getArray(0);
    }

    Expr RLEModeFormat::getVar(std::string postfix, Mode mode, Datatype t) const {
      const std::string varName = mode.getName() + postfix;

      if (!mode.hasVar(varName)) {
        Expr idxCapacity = Var::make(varName, t);
        mode.addVar(varName, idxCapacity);
        return idxCapacity;
      }

      return mode.getVar(varName);
    }


    Expr RLEModeFormat::getCoordVar(Mode mode) const {
      return getVar("_coord", mode);
    }

    Expr RLEModeFormat::getPosCoordVar(Mode mode) const {
      return getVar("_pos_coord", mode);
    }

    Expr RLEModeFormat::getFoundVar(Mode mode) const {
      return getVar("_found", mode);
    }

    Expr RLEModeFormat::getFoundCountVar(Mode mode) const {
      return getVar("_found_cnt", mode);
    }

    ir::Expr RLEModeFormat::getFillingBoolVar(Mode mode) const {
      return getVar("_is_filling", mode, Bool);
    }

    ir::Expr RLEModeFormat::getCurrCountVar(Mode mode) const {
      return getVar("_cnt_pos", mode);
    }

    ir::Expr RLEModeFormat::getCurrCountValVar(Mode mode) const {
      return getVar("_cnt_val", mode);
    }

    Expr RLEModeFormat::getPosCapacity(Mode mode) const {
      return getVar("_pos_capacity", mode);
    }

    Expr RLEModeFormat::getWidth(Mode mode) const {
      return ir::Literal::make(allocSize, Datatype::Int32);
    }

    bool RLEModeFormat::equals(const ModeFormatImpl& other) const {
      return ModeFormatImpl::equals(other) &&
             (dynamic_cast<const RLEModeFormat&>(other).allocSize == allocSize);
    }

}
